_base_ = [
    '../../_base_/models/mask_rcnn_r50_fpn.py',
    '../../_base_/datasets/uav_video_dataset_mask.py', '../../_base_/default_runtime.py'
]

img_scale = (2160, 4096)
dataset_type = 'UAVVIDEODATASET'

model = dict(
    type='ByteTrack',
    detector=dict(
        rpn_head=dict(bbox_coder=dict(clip_border=False)),
        roi_head=dict(
            bbox_head=dict(bbox_coder=dict(clip_border=False), num_classes=1)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            r"C:\Users\jac41744\mmtracking\uav_ext\adam0001\4\work_dirs\mask-rcnn_r50_fpn_4e_uav\epoch_7.pth"
        )),
    motion=dict(type='KalmanFilter', center_only=True),
    tracker=dict(
        type='ByteTracker',
        obj_score_thrs=dict(high=0.6, low=0.05),
        init_track_thr=0.6,
        weight_iou_with_det_scores=True,  # change maskrcnn nms
        match_iou_thrs=dict(high=0.035, low=0.25, tentative=0.085),
        num_tentatives=5,
        num_frames_retain=50))

# train_pipeline = [
#     dict(
#         type='Mosaic',
#         img_scale=img_scale,
#         pad_val=114.0,
#         bbox_clip_border=False),
#     dict(
#         type='RandomAffine',
#         scaling_ratio_range=(0.1, 2),
#         border=(-img_scale[0] // 2, -img_scale[1] // 2),
#         bbox_clip_border=False),
#     dict(
#         type='MixUp',
#         img_scale=img_scale,
#         ratio_range=(0.8, 1.6),
#         pad_val=114.0,
#         bbox_clip_border=False),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Pad', size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0))),
#     dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
#     dict(type='DefaultFormatBundle'),
#     dict(
#         type='VideoCollect',
#         keys=[
#             'img', 'gt_bboxes', 'gt_labels', 'gt_match_indices',
#             'gt_instance_ids', 'gt_masks'
#         ]),
# ]
#
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=img_scale,
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(
#                 type='Pad',
#                 size_divisor=32,
#                 pad_val=dict(img=(114.0, 114.0, 114.0))),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='VideoCollect', keys=['img'])
#         ])
# ]
# data_root = "C:\\Users\\jac41744\\smartvisiontoolbox\\examples\\data\\uav_cocovid/"
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         visibility_thr=-1,
#         ann_file=data_root + 'train/coco_train.json',
#         img_prefix=data_root + 'train',
#         ref_img_sampler=dict(
#             num_ref_imgs=1,
#             frame_range=10,
#             filter_key_img=True,
#             method='uniform'),
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'test/coco_test.json',
#         img_prefix=data_root + 'test',
#         ref_img_sampler=None,
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'test/coco_test.json',
#         img_prefix=data_root + 'test',
#         ref_img_sampler=None,
#         pipeline=test_pipeline))


# optimizer
# default 8 gpu
# optimizer = dict(
#     type='SGD',
#     lr=0.001 / 8 * samples_per_gpu,
#     momentum=0.9,
#     weight_decay=5e-4,
#     nesterov=True,
#     paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
# optimizer_config = dict(grad_clip=None)

# some hyper parameters
total_epochs = 20
# num_last_epochs = 10
# resume_from = None
# interval = 5

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 100,
    step=[5])

checkpoint_config = dict(interval=1)
evaluation = dict(metric=['bbox', 'segm', 'track', 'track_segm'], interval=1)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
# fp16 = dict(loss_scale=dict(init_scale=512.))
