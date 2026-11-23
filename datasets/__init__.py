"""Dataset registrations used by the evaluation configs."""

from .custom import (COCOObjectDataset, PascalContext59Dataset,
                     PascalContext60Dataset, PascalVOC20Dataset)

__all__ = [
    'COCOObjectDataset', 'PascalContext59Dataset', 'PascalContext60Dataset',
    'PascalVOC20Dataset'
]
