# dataset settings
dataset_type = 'CocoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=True,),
    # dict(
    #     type='Resize',
    #     img_scale=(1088, 1088),
    #     ratio_range=(0.8, 1.2),
    #     keep_ratio=True,
    #     bbox_clip_border=False),
    dict(type='PhotoMetricDistortion'),
    # dict(type='RandomCrop', crop_size=(1088, 1088), bbox_clip_border=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2160, 4096),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data_root = "C:\\Users\\jac41744\\smartvisiontoolbox\\examples\\data\\uav_cocovid/"
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/coco_train.json',
        img_prefix=data_root + 'train',
        classes=('grape', ),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/coco_val.json',
        img_prefix=data_root + 'val',
        classes=('grape', ),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/coco_test.json',
        img_prefix=data_root + 'test',
        classes=('grape', ),
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric=['bbox', 'segm'])
