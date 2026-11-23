_base_ = '../_base_/evaluation.py'
from pathlib import Path


# model settings
model = dict(
    name_path=str(Path(__file__).resolve().parents[1] / 'classes' / 'voc21.txt'),
    instance_mask_path='data/region_masks/voc',
    prob_thd=0.1,
)

# dataset settings
dataset_type = 'PascalVOCDataset'
data_root = 'data/VOCdevkit/VOC2012'

test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
    ),
)
