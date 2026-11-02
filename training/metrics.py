"""Per-slice metrics following the recovered evaluation protocol."""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_map_metrics(prediction, reference, mask=None, physical_scale=1.0):
    prediction = np.clip(np.asarray(prediction, dtype=np.float64), 0, 1)
    reference = np.clip(np.asarray(reference, dtype=np.float64), 0, 1)
    if prediction.ndim != 2 or prediction.shape != reference.shape:
        raise ValueError('Metrics require matching 2D maps')
    roi = reference > 0 if mask is None else np.asarray(mask) > 0
    if roi.shape != reference.shape or not roi.any():
        raise ValueError('Evaluation mask must match the map and contain foreground')
    ref_values = reference[roi] * physical_scale
    pred_values = prediction[roi] * physical_scale
    masked_reference, masked_prediction = reference * roi, prediction * roi
    minimum_size = min(reference.shape)
    window = min(7, minimum_size if minimum_size % 2 else minimum_size - 1)
    if window < 3:
        raise ValueError('SSIM requires images at least 3 pixels wide and tall')
    return {
        'nmse': float(np.sum((pred_values-ref_values)**2) / (np.sum(ref_values**2)+1e-8)),
        'mae': float(np.mean(np.abs(pred_values-ref_values))),
        'psnr': float(peak_signal_noise_ratio(masked_reference, masked_prediction, data_range=1.0)),
        'ssim': float(structural_similarity(masked_reference, masked_prediction, data_range=1.0, win_size=window)),
    }
