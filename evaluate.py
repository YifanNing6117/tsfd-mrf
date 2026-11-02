"""Evaluate both TSFD parameter maps with per-slice and aggregate metrics."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from data.dataset import create_data_loader
from training.engine import build_tsfd_model, load_checkpoint
from training.metrics import compute_map_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--dataset', choices=['brainweb', 'invivo', 'delics'], required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--frames', type=int, choices=[200, 400], default=400)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output', default='runs/tsfd/evaluation')
    args = parser.parse_args()
    device = torch.device(args.device)
    model = build_tsfd_model(device)
    load_checkpoint(args.checkpoint, model, device)
    model.eval()
    loader = create_data_loader(args.manifest, args.dataset, args.split,
                                args.frames, args.data_root, workers=args.workers)
    rows = []
    with torch.no_grad():
        for batch in loader:
            predictions = model(batch['inputs'].to(device)).float().cpu().numpy()[0]
            reference = batch['targets'].numpy()[0]
            for channel, modality in enumerate(['T1', 'T2']):
                scale = [5000., 4000.][channel] if args.dataset == 'delics' else 1.0
                mask = batch['mask'].numpy()[0, 0] if args.dataset == 'delics' else None
                values = compute_map_metrics(predictions[channel], reference[channel], mask, scale)
                rows.append(dict(sample=batch['path'][0], slice_index=int(batch['slice_index'][0]),
                                 modality=modality, **values,
                                 mae_unit='ms' if args.dataset == 'delics' else 'normalized'))
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    with (output/'per_slice_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {'dataset': args.dataset, 'split': args.split, 'frames': args.frames, 'samples': len(rows)//2,
               'mae_unit': rows[0]['mae_unit'], 'modalities': {}}
    for modality in ['T1', 'T2']:
        selected = [row for row in rows if row['modality'] == modality]
        summary['modalities'][modality] = {
            name: {'mean': float(np.mean([row[name] for row in selected])),
                   'std': float(np.std([row[name] for row in selected], ddof=0))}
            for name in ['psnr', 'ssim', 'nmse', 'mae']}
    # Perfect predictions can produce infinite PSNR; represent those values as strings.
    for metrics in summary['modalities'].values():
        for metric in metrics.values():
            for key, value in metric.items():
                if not np.isfinite(value):metric[key] = str(value)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
