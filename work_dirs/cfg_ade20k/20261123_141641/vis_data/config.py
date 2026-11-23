data_root = '~/Documents/dataset/ADE/ADEChallengeData2016/'
dataset_type = 'ADE20KDataset'
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    clip_type='openai',
    dino_type='dino_vitb8',
    instance_mask_path='data/region_masks/ade',
    model_type='ViT-B/16',
    name_path=
    '/mnt/data0/zhangbaiqi/Documents/root/sticther/configs/datasets/../classes/ade20k.txt',
    slide_crop=336,
    slide_stride=112,
    template_path=
    '/mnt/data0/zhangbaiqi/Documents/root/sticther/configs/_base_/../../prompts/class_biased_template.json',
    type='OVStitcherSegmentation')
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        data_root='~/Documents/dataset/ADE/ADEChallengeData2016/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                448,
            ), type='Resize'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='ADE20KDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2048,
        448,
    ), type='Resize'),
    dict(reduce_zero_label=True, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
work_dir = 'work_dirs/cfg_ade20k'
