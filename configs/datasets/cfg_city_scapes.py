_base_ = '../_base_/evaluation.py'


# model settings
model = dict(
    name_path='{{fileDirname}}/../classes/cityscapes.txt',
    instance_mask_path='data/region_masks/city',
    slide_crop=224,
)

# dataset settings
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 560), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=test_pipeline,
    ),
)
