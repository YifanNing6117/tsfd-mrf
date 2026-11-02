# TSFD

Temporal-Spatial Feature-Disentangled Network for magnetic resonance fingerprinting (MRF). This repository contains the model definition and result tables reported in the TSFD-MRF manuscript.

## Model

Install a PyTorch build compatible with your CUDA environment, followed by the dependencies in `requirements.txt`. The Mamba selective-scan operator requires a compatible runtime.

```python
from model.tsfd import TSFD

model = TSFD(num_out_chans=2, img_size=256)
```

The input shape is `[B, 400, 256, 256]`. The first convolution accepts 400 channels, including when `in_chans` has another value. Shorter sequences require the preprocessing and padding used in the original experiments. This repository provides the network definition; training and data-processing scripts remain in the local project archive.

The Temporal Spectral Mamba (TSM) block combines Pixel-wise Temporal Mamba (PT-Mamba) and Global Spectral Filter (GSF) outputs through a sigmoid gate, then adds the input residual. The Cross-Interaction Fusion Module (CIFM) adds a shared interaction feature to each stream before channel projection.

## Manuscript Results

The images below reproduce the methods, numerical precision, standard deviations, and bold or underlined entries from the supplied manuscript tables. The in vivo and Deli-CS sections are displayed separately. Sequence-length groups are stacked vertically for readability.

These are manuscript-reported values. Some differ from the original experiment TXT records. The TXT-derived CSV files remain in the local archive. No metrics were recomputed, and no training or GPU inference was performed during repository preparation.

### BrainWeb

![BrainWeb comparison](images/brainweb_comparison.png)

### In Vivo 3D MRF

![In vivo comparison](images/invivo_comparison.png)

### Deli-CS

![Deli-CS comparison](images/delics_comparison.png)

### Model Complexity

![Model complexity](images/model_complexity.png)

### Component Ablation

![Component ablation](images/component_ablation.png)

## Files

| Path | Contents |
| --- | --- |
| `model/tsfd.py` | TSFD network definition |
| `images/` | Five manuscript result-table images |
| `requirements.txt` | Model dependencies |
