#!/usr/bin/env python3
import argparse
import datetime as dt
import re
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PYTHON = "/work/envs/ningyi_clone/bin/python"
CHECKPOINTS_DIR = "EXP_checkpoints"
DATAROOT = "delics_2min_dataset.csv"
RESULT_ROOT = Path("ablation_outputs")
MAIN_MODEL = "delics_2min_50DWTrealtfmambampu_v5_nokaiming_lr8e-4_lr1e-6"
MAIN_NET = "DWTtfmambampu_v5"
ABLATIONS = [
    ("no_lwe", "No LWE", "DWTtfmambampu_v5_no_lwe"),
    ("no_gsf", "No GSF", "DWTtfmambampu_v5_no_gsf"),
    ("no_cifm", "No CIFM", "DWTtfmambampu_v5_no_cifm"),
    ("no_swin", "No Swin", "DWTtfmambampu_v5_no_swin"),
    ("no_ptmamba", "No PTmamba", "DWTtfmambampu_v5_no_ptmamba"),
]
TEST_MODALITIES = ("t1", "t2")
METRIC_RE = re.compile(r"^(PSNR|SSIM|NMSE|MAE \\(ms\\)):\\s*([0-9.]+)\\s*\\+/-\\s*([0-9.]+)", re.M)


def run_logged(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("COMMAND: " + " ".join(cmd) + "\\n\\n")
        f.flush()
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)


def wait_until_below_limit(running, limit):
    while len(running) >= limit:
        for item in list(running):
            ret = item["proc"].poll()
            if ret is None:
                continue
            running.remove(item)
            print("[train done] {} gpu={} returncode={}".format(item["name"], item["gpu"], ret), flush=True)
            if ret != 0:
                raise RuntimeError("Training failed for {}. See {}".format(item["name"], item["log"]))
        if len(running) >= limit:
            time.sleep(20)


def train_ablations(args):
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    running = []
    for idx, (suffix, _label, net) in enumerate(ABLATIONS):
        wait_until_below_limit(running, len(gpus))
        gpu = gpus[idx % len(gpus)]
        exp = f"ablation_2min_{suffix}_DWTtfmambampu_v5_lr8e-4_lr1e-6"
        log = RESULT_ROOT / "logs" / f"train_{exp}.log"
        cmd = [
            PYTHON, "train_wyz.py",
            "--dataroot", args.dataroot,
            "--name", exp,
            "--checkpoints_dir", CHECKPOINTS_DIR,
            "--gpu_ids", gpu,
            "--dataset_mode", "delics",
            "--model", "resvit_one",
            "--which_model_netG", net,
            "--input_nc", "400",
            "--output_nc", "2",
            "--norm", "none",
            "--fineSize", "256",
            "--loadSize", "256",
            "--resize_or_crop", "none",
            "--batchSize", "1",
            "--nThreads", str(args.threads),
            "--niter", str(args.epochs),
            "--niter_decay", "0",
            "--lr", str(args.lr),
            "--beta1", "0.9",
            "--save_epoch_freq", "1",
            "--save_latest_freq", "5000",
            "--display_id", "0",
            "--pre_trained_resnet", "0",
            "--pre_trained_transformer", "0",
            "--mask", "G1D30",
        ]
        print(f"[train start] {exp} net={net} gpu={gpu}", flush=True)
        running.append({"proc": run_logged(cmd, log), "name": exp, "gpu": gpu, "log": log})
    while running:
        wait_until_below_limit(running, 1)


def run_test(model, net, modality, gpu, args):
    out = RESULT_ROOT / "tests" / model / modality
    log = RESULT_ROOT / "logs" / f"test_{model}_{modality}.log"
    cmd = [
        PYTHON, "test_delics.py",
        "--name", model,
        "--checkpoints_dir", CHECKPOINTS_DIR,
        "--gpu_ids", gpu,
        "--dataroot", args.dataroot,
        "--how_many", str(args.how_many),
        "--results_dir", str(out),
        "--modality", modality,
        "--model", "resvit_one",
        "--which_model_netG", net,
        "--dataset_mode", "delics",
        "--norm", "none",
        "--input_nc", "400",
        "--output_nc", "2",
        "--serial_batches",
        "--fineSize", "256",
        "--loadSize", "256",
        "--which_epoch", "latest",
        "--pre_trained_resnet", "0",
        "--pre_trained_transformer", "0",
    ]
    print(f"[test start] {model} {modality} gpu={gpu}", flush=True)
    proc = run_logged(cmd, log)
    ret = proc.wait()
    print(f"[test done] {model} {modality} returncode={ret}", flush=True)
    if ret != 0:
        raise RuntimeError(f"Test failed for {model} {modality}. See {log}")
    return out


def parse_metrics(path):
    text = path.read_text(encoding="utf-8")
    data = {}
    for name, mean, std in METRIC_RE.findall(text):
        key = "MAE" if name.startswith("MAE") else name
        data[key] = (float(mean), float(std))
    return data


def collect(models):
    rows = []
    for model, label, net in models:
        row = {"model": model, "label": label, "net": net}
        for mod in TEST_MODALITIES:
            mp = RESULT_ROOT / "tests" / model / mod / "metrics.txt"
            if mp.exists():
                for metric, (mean, std) in parse_metrics(mp).items():
                    row[f"{mod}_{metric}_mean"] = mean
                    row[f"{mod}_{metric}_std"] = std
        rows.append(row)
    return rows


def write_csv(rows):
    keys = ["model", "label", "net"]
    for mod in TEST_MODALITIES:
        for metric in ("PSNR", "SSIM", "NMSE", "MAE"):
            keys += [f"{mod}_{metric}_mean", f"{mod}_{metric}_std"]
    path = RESULT_ROOT / "ablation_metrics.csv"
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\\n")
    return path


def plot_metric(rows, mod, metric):
    valid = [r for r in rows if f"{mod}_{metric}_mean" in r]
    if not valid:
        return None
    labels = [r["label"] for r in valid]
    means = [r[f"{mod}_{metric}_mean"] for r in valid]
    stds = [r.get(f"{mod}_{metric}_std", 0.0) for r in valid]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=180)
    ax.bar(np.arange(len(means)), means, yerr=stds, color=["#3b6ea8"] + ["#8f8f8f"] * (len(means) - 1), capsize=3, width=0.68)
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"Deli-CS {mod.upper()} {metric} Ablation")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = RESULT_ROOT / "figures" / f"ablation_{mod}_{metric.lower()}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def test_and_plot(args):
    models = [(MAIN_MODEL, "Full", MAIN_NET)]
    models += [(f"ablation_2min_{s}_DWTtfmambampu_v5_lr8e-4_lr1e-6", label, net) for s, label, net in ABLATIONS]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    for idx, (model, _label, net) in enumerate(models):
        for mod in TEST_MODALITIES:
            run_test(model, net, mod, gpus[idx % len(gpus)], args)
    rows = collect(models)
    print(f"[metrics] {write_csv(rows)}", flush=True)
    for mod in TEST_MODALITIES:
        for metric in ("PSNR", "SSIM", "NMSE", "MAE"):
            out = plot_metric(rows, mod, metric)
            if out:
                print(f"[figure] {out}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--dataroot", default=DATAROOT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--how_many", type=int, default=100)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    args = parser.parse_args()
    RESULT_ROOT.mkdir(exist_ok=True)
    print(f"[start] {dt.datetime.now().isoformat(timespec=seconds)}", flush=True)
    if not args.skip_train:
        train_ablations(args)
    if not args.skip_test:
        test_and_plot(args)
    print(f"[done] {dt.datetime.now().isoformat(timespec=seconds)}", flush=True)


if __name__ == "__main__":
    main()
