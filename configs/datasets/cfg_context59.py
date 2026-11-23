_base_ = '../_base_/evaluation.py'


# model settings
model = dict(
    name_path='{{fileDirname}}/../classes/context59.txt',
    instance_mask_path='data/region_masks/context',
)

# dataset settings
dataset_type = 'PascalContext59Dataset'
data_root = 'data/VOCdevkit/VOC2010'

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
        data_prefix=dict(img_path='JPEGImages',
                         seg_map_path='SegmentationClassContext'),
        ann_file='ImageSets/SegmentationContext/val.txt',
        pipeline=test_pipeline,
    ),
)
