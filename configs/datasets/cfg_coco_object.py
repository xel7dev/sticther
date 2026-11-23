_base_ = '../_base_/evaluation.py'


# model settings
model = dict(
    name_path='{{fileDirname}}/../classes/coco_object.txt',
    instance_mask_path='data/region_masks/coco',
    prob_thd=0.2,
)

# dataset settings
dataset_type = 'COCOObjectDataset'
data_root = 'data/coco_object/'

test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(img_path='images/val2017',
                         seg_map_path='annotations/val2017'),
    ),
)
