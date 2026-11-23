_base_ = '../_base_/evaluation.py'
from pathlib import Path


# model settings
model = dict(
    name_path=str(Path(__file__).resolve().parents[1] / 'classes' / 'ade20k.txt'),
    instance_mask_path='data/region_masks/ade',
)

# dataset settings
dataset_type = 'ADE20KDataset'
data_root = 'data/ADEChallengeData2016/'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 448), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/validation',
                         seg_map_path='annotations/validation'),
        pipeline=test_pipeline,
    ),
)
