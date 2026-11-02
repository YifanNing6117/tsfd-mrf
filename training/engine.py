"""TSFD optimization, validation and resumable checkpoints."""
from pathlib import Path

import torch

from training.loss import TSFDLoss


def build_tsfd_model(device):
    from model import tsfd
    if tsfd.selective_scan_fn is None:
        raise RuntimeError('Install a CUDA-compatible mamba-ssm selective-scan kernel before running TSFD')
    return tsfd.TSFD(num_out_chans=2, img_size=256, use_checkpoint=True).to(device)


def create_optimizer(model, learning_rate=8e-4, minimum_lr=1e-7, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 betas=(0.9, 0.999), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=minimum_lr)
    return optimizer, scheduler


def run_epoch(model, loader, device, optimizer=None, scaler=None,
              amp=False, mask_loss=False):
    training = optimizer is not None
    model.train(training)
    criterion = TSFDLoss()
    total_loss, samples = 0.0, 0
    if amp and device.type != 'cuda':
        raise ValueError('AMP requires a CUDA device')
    with torch.set_grad_enabled(training):
        for batch in loader:
            inputs = batch['inputs'].to(device, non_blocking=True)
            targets = batch['targets'].to(device, non_blocking=True)
            mask = batch['mask'].to(device) if mask_loss else None
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                predictions = model(inputs)
                loss = criterion(predictions, targets, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite TSFD loss')
            if training:
                if amp:
                    if scaler is None:
                        raise ValueError('AMP training requires a GradScaler')
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            size = inputs.shape[0]
            total_loss += loss.detach().item() * size
            samples += size
    if samples == 0:
        raise ValueError('Data loader is empty')
    return total_loss / samples


def save_checkpoint(path, model, optimizer, scheduler, epoch, config,
                    scaler=None, best_loss=float('inf')):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'epoch': epoch,
                'config': config, 'best_loss': best_loss,
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'scaler': scaler.state_dict() if scaler is not None else None}, path)


def load_checkpoint(path, model, device, optimizer=None, scheduler=None, scaler=None):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint['model'] if 'model' in checkpoint else checkpoint
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    for name, component in [('optimizer', optimizer), ('scheduler', scheduler), ('scaler', scaler)]:
        if component is not None:
            if name not in checkpoint or checkpoint[name] is None:
                raise ValueError('Resume requires checkpoint state: ' + name)
            component.load_state_dict(checkpoint[name])
    if optimizer is not None and 'rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['rng_state'].cpu())
        if torch.cuda.is_available() and checkpoint.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['cuda_rng_state']])
    return checkpoint
