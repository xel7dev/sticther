_base_ = '../_base_/evaluation.py'


# model settings
model = dict(
    name_path='{{fileDirname}}/../classes/context60.txt',
    instance_mask_path='data/region_masks/context',
    prob_thd=0.1,
)

# dataset settings
dataset_type = 'PascalContext60Dataset'
data_root = 'data/VOCdevkit/VOC2010'

test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='JPEGImages',
                         seg_map_path='SegmentationClassContext'),
        ann_file='ImageSets/SegmentationContext/val.txt',
    ),
)
