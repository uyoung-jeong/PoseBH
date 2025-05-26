_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/pw3d.py'
]
checkpoint_config = dict(interval=5)
evaluation = dict(interval=5, metric='mAP', save_best='AP')

optimizer = dict(type='Adam',lr=1e-4,)

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[10, 15])
total_epochs = 30
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=24,
    dataset_joints=24,
    dataset_channel=[
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23
        ],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23
    ])

num_in_class_proto = 3

# custom hooks
custom_hooks =[
    dict(
        type='EmbFreezeThawHook',
        freeze=['freeze,0'],
    ),
    dict(
        type='ProtoFreezeThawHook',
        freeze=['thaw,0', 'freeze,15'],
    ),
    dict(
        type='BackboneFreezeHook',
        freeze=['freeze,0', 'thaw,15', 'freeze,16']
    )
]

# model settings
model = dict(
    type='TopDownProto',
    pretrained=None,
    multihead_pretrained='work_dirs/vitb_posebh/coco_no_proto.pth',
    backbone=dict(
        type='ViT',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=768,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1, ),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    proto_head=dict(
        type='KptProtoHead',
        in_channels=768,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(
            channels=64,
            head_kernel=3,
            activation='silu',
            neck_type='res',
            num_keypoints=[24],
            num_in_class_proto=num_in_class_proto,
            hard_cluster=True,
            cluster_conf_thr=0.2,
            gamma=0.999,
            loss_proto=dict(type='PixelPrototypeCELoss',
                            cfg=dict(weight=1.25e-5,
                                    ppc_weight=0.01,
                                    ppd_weight=0.001,
                                    num_in_class_proto=num_in_class_proto,
                                    num_joints=214)),
            phm_loss_weight=1.25e-5,
            css_weight=0.0,
            css_conf_thr=0.25,
            css_update_weight=3.13e-9,
            css_match_dist=2.1),
        out_channels=64,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='./data/coco/person_detection_results/'
    'COCO_val2017_detections_AP_H_56_person.json',
    max_num_joints=24,
    dataset_idx=0,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'dataset_idx'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs', 'dataset_idx'
        ]),
]

test_pipeline = val_pipeline

data_root = './data/pw3d'
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=[dict(
        type='TopDownPW3DDataset',
        ann_file='data/pw3d/annotations/pw3d_train.json',
        img_prefix=f'{data_root}/imageFiles/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}})],
    val=dict(
        type='TopDownPW3DDataset',
        ann_file=f'{data_root}/annotations/pw3d_test.json',
        img_prefix=f'{data_root}/imageFiles/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownPW3DDataset',
        ann_file=f'{data_root}/annotations/pw3d_test.json',
        img_prefix=f'{data_root}/imageFiles/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
