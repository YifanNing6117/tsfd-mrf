"""Train the TSFD model using the manuscript's optimization defaults."""
import argparse
import csv
import json
from pathlib import Path

import torch

from data.dataset import create_data_loader
from training.engine import build_tsfd_model, create_optimizer, load_checkpoint, run_epoch, save_checkpoint


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--dataset', choices=['brainweb', 'invivo', 'delics'], required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--frames', type=int, choices=[200, 400], default=400)
    parser.add_argument('--train-split', default='train')
    parser.add_argument('--validation-split', default='valid')
    parser.add_argument('--output', default='runs/tsfd')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=8e-4)
    parser.add_argument('--minimum-lr', type=float, default=1e-7)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--mask-loss', action='store_true', help='Use the Deli-CS wrapper masked-L1 variant')
    parser.add_argument('--resume')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError('Invalid epoch, batch-size or worker setting')
    if args.train_split == args.validation_split:
        raise ValueError('Training and validation splits must differ')
    if args.mask_loss and args.dataset != 'delics':
        raise ValueError('--mask-loss is specific to Deli-CS')
    if args.amp and not args.device.startswith('cuda'):
        raise ValueError('--amp requires a CUDA device')
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = build_tsfd_model(device)
    optimizer, scheduler = create_optimizer(model, args.learning_rate, args.minimum_lr, args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    common = dict(manifest=args.manifest, dataset=args.dataset, frames=args.frames,
                  data_root=args.data_root, batch_size=args.batch_size, workers=args.workers)
    train_loader = create_data_loader(split=args.train_split, shuffle=True, **common)
    valid_loader = create_data_loader(split=args.validation_split, **common)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    start, best_loss = 1, float('inf')
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, device, optimizer, scheduler, scaler)
        previous = checkpoint.get('config', {})
        for key in ['dataset', 'frames', 'epochs', 'learning_rate', 'minimum_lr', 'mask_loss', 'amp',
                    'manifest', 'data_root', 'train_split', 'validation_split', 'batch_size', 'seed']:
            if previous.get(key) != vars(args)[key]:
                raise ValueError('Resume setting differs: ' + key)
        start, best_loss = checkpoint['epoch']+1, checkpoint.get('best_loss', best_loss)
    (output/'config.json').write_text(json.dumps(vars(args), indent=2)+'\n', encoding='utf-8')
    history = output/'loss_history.csv'
    with history.open('a' if args.resume else 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'learning_rate', 'train_l1', 'validation_l1'])
        if stream.tell() == 0:
            writer.writeheader()
        for epoch in range(start, args.epochs+1):
            rate = optimizer.param_groups[0]['lr']
            train_loss = run_epoch(model, train_loader, device, optimizer, scaler, args.amp, args.mask_loss)
            valid_loss = run_epoch(model, valid_loader, device, amp=args.amp, mask_loss=args.mask_loss)
            improved = valid_loss < best_loss
            best_loss = min(best_loss, valid_loss)
            scheduler.step()
            state = dict(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch,
                         config=vars(args), scaler=scaler, best_loss=best_loss)
            save_checkpoint(output/'latest.pt', **state)
            if improved:
                save_checkpoint(output/'best.pt', **state)
            row = dict(epoch=epoch, learning_rate=rate, train_l1=train_loss, validation_l1=valid_loss)
            writer.writerow(row); stream.flush()
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
