"""Precomputed region masks and region-level prediction correction."""

from pathlib import Path

import numpy as np
import torch


def load_instance_mask(mask_dir, image_path, device):
    """Load one ``[Horig,Worig]`` integer region-ID map from ``.npz``."""
    path = Path(mask_dir) / f'{Path(image_path).stem}.npz'
    if not path.is_file():
        raise FileNotFoundError(f'Region mask missing: {path}')
    with np.load(path) as archive:
        if 'instance_mask' not in archive:
            raise KeyError(f'instance_mask missing in {path}')
        mask = torch.from_numpy(archive['instance_mask']).to(device)
    if mask.ndim != 2:
        raise ValueError(
            f'Expected a 2D instance_mask in {path}, got {tuple(mask.shape)}')
    return mask


def correct_predictions_by_region(prediction, instance_mask):
    """Replace each non-background region with its majority class.

    ``prediction`` is ``[1,Horig,Worig]`` and ``instance_mask`` is
    ``[Horig,Worig]``. Region ID 0 is reserved for unsegmented/background
    pixels; each other connected mask receives its most common predicted class
    to suppress patch-level label noise inside a coherent object region.
    """
    for region_id in torch.unique(instance_mask)[1:]:
        region = instance_mask == region_id
        prediction[:, region] = torch.mode(prediction[:, region].flatten())[0]
    return prediction
