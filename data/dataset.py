"""Slice loading and preprocessing for the TSFD experiments."""
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


def read_image(path):
    """Read a NIfTI image in TorchIO's channel-first convention."""
    import torchio
    return torchio.ScalarImage(str(path)).data.float()


def normalize_minmax(image):
    return (image - image.min()) / (image.max() - image.min() + 1e-8)


def preprocess_slice(fingerprint, t1, t2, dataset, frames=400, mask=None):
    """Preserve the two preprocessing paths from the recovered loaders."""
    if dataset not in {'brainweb', 'invivo', 'delics'}:
        raise ValueError('Unsupported dataset: ' + dataset)
    if not 1 <= frames <= 400:
        raise ValueError('frames must be between 1 and 400')
    if dataset == 'delics':
        if fingerprint.ndim != 4 or fingerprint.shape[1] != 1:
            raise ValueError('Deli-CS input must have TorchIO shape [T, 1, H, W]')
        inputs = fingerprint[:frames, 0].clone()
        if t1.shape[:2] != (1, 1) or t2.shape[:2] != (1, 1):
            raise ValueError('Deli-CS targets must have shape [1, 1, H, W]')
        target_t1 = (t1[0, 0].unsqueeze(0) / 5000.0).clamp(0, 1)
        target_t2 = (t2[0, 0].unsqueeze(0) / 4000.0).clamp(0, 1)
        if mask is None or mask.shape[:2] != (1, 1):
            raise ValueError('Deli-CS requires a mask with shape [1, 1, H, W]')
        roi = mask[0, 0].unsqueeze(0).clone()
        inputs = inputs / (inputs.max() + 1e-8)
    else:
        if fingerprint.ndim != 4 or fingerprint.shape[0] != 1:
            raise ValueError('BrainWeb/in vivo input must have shape [1, H, W, T]')
        inputs = normalize_minmax(fingerprint[0, :, :, :frames].permute(2, 0, 1).contiguous())
        if t1.ndim != 4 or t1.shape[0] != 1 or t1.shape[-1] != 1 or t2.shape != t1.shape:
            raise ValueError('BrainWeb/in vivo targets must have shape [1, H, W, 1]')
        target_t1 = normalize_minmax(t1.squeeze(-1))
        target_t2 = normalize_minmax(t2.squeeze(-1))
        roi = torch.ones_like(target_t1)
    if inputs.shape[0] != frames:
        raise ValueError('Input has fewer frames than requested')
    targets = torch.cat((target_t1, target_t2), dim=0)
    if inputs.shape[1:] != targets.shape[1:] or roi.shape != target_t1.shape:
        raise ValueError('Input, target and mask spatial shapes disagree')
    if not all(torch.isfinite(x).all() for x in (inputs, targets, roi)):
        raise ValueError('Input, targets and mask must contain finite values')
    padded = inputs.new_zeros((400, *inputs.shape[1:]))
    padded[:frames] = inputs
    return padded.float(), targets.float(), roi.float()


class TSFDDataset(Dataset):
    """Load pre-extracted slices, filtering the manifest by dataset split."""
    def __init__(self, manifest, dataset, split, frames=400, data_root=None):
        self.manifest = Path(manifest).resolve()
        self.root = Path(data_root).resolve() if data_root else self.manifest.parent
        self.dataset, self.frames = dataset, frames
        with self.manifest.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.rows = [row for row in rows if row.get('split', row.get('fold')) == split]
        if not self.rows:
            raise ValueError('No samples for split ' + split)

    def __len__(self):
        return len(self.rows)

    def _path(self, row, *names):
        value = next((row[n] for n in names if row.get(n)), None)
        if value is None:
            raise ValueError('Manifest requires column ' + names[0])
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def __getitem__(self, index):
        row = self.rows[index]
        input_path = self._path(row, 'input_path', 'timsmask400')
        t1_path = self._path(row, 't1_path', 't1')
        t2_path = self._path(row, 't2_path', 't2')
        mask = read_image(self._path(row, 'mask_path')) if self.dataset == 'delics' else None
        inputs, targets, roi = preprocess_slice(
            read_image(input_path), read_image(t1_path), read_image(t2_path),
            self.dataset, self.frames, mask,
        )
        return {'inputs': inputs, 'targets': targets, 'mask': roi,
                'path': str(input_path),
                'slice_index': int(row.get('slice_index', row.get('slice_idx', row.get('slice_indices', 0))))}


def create_data_loader(manifest, dataset, split, frames=400, data_root=None,
                       batch_size=1, workers=2, shuffle=False):
    slices = TSFDDataset(manifest, dataset, split, frames, data_root)
    return DataLoader(slices, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=torch.cuda.is_available())
