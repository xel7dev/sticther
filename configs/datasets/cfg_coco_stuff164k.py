_base_ = '../_base_/evaluation.py'


# model settings
model = dict(
    name_path='{{fileDirname}}/../classes/coco_stuff.txt',
    instance_mask_path='data/region_masks/coco',
    slide_stride=224,
)

# dataset settings
dataset_type = 'COCOStuffDataset'
data_root = 'data/coco_stuff164k'

test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/val2017',
                         seg_map_path='annotations/val2017'),
    ),
)
