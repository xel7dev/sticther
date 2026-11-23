_base_ = '../_base_/evaluation.py'


# model settings
model = dict(
    name_path='{{fileDirname}}/../classes/voc20.txt',
    instance_mask_path='data/region_masks/voc',
)

# dataset settings
dataset_type = 'PascalVOC20Dataset'
data_root = 'data/VOCdevkit/VOC2012'

test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
    ),
)
