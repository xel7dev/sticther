from pathlib import Path


# Shared settings for the eight evaluation datasets.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
model = dict(
    type='OVStitcherSegmentation',
    clip_type='openai',
    model_type='ViT-B/16',
    dino_type='dino_vitb8',
    template_path=str(PACKAGE_ROOT / 'prompts' / 'class_biased_template.json'),
    slide_stride=112,
    slide_crop=336,
)

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_cfg = dict(type='TestLoop')

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 448), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(pipeline=test_pipeline),
)

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_processor = dict(by_epoch=False)
log_level = 'INFO'
