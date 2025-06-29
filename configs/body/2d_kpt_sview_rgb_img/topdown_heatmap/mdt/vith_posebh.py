_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/coco.py',
    '../../../../_base_/datasets/aic_info.py',
    '../../../../_base_/datasets/mpii_info.py',
    '../../../../_base_/datasets/ap10k_info.py',
    '../../../../_base_/datasets/coco_wholebody_info.py'
]
evaluation = dict(interval=10, metric='mAP', save_best='AP')

optimizer = dict(type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.1,
                 constructor='LayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(
                                    num_layers=32, 
                                    layer_decay_rate=0.8,
                                    custom_keys={
                                            'bias': dict(decay_multi=0.),
                                            'pos_embed': dict(decay_mult=0.),
                                            'relative_position_bias_table': dict(decay_mult=0.),
                                            'norm': dict(decay_mult=0.)
                                            }
                                    )
                )

optimizer_config = dict(grad_clip=dict(max_norm=1., norm_type=2))
fp16 = dict(loss_scale=dict(
                    init_scale=16384.0
                )
            )

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[50, 90])
total_epochs = 100
target_type = 'GaussianHeatmap'
channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])
aic_channel_cfg = dict(
    num_output_channels=14,
    dataset_joints=14,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    ],
    inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
mpii_channel_cfg = dict(
    num_output_channels=16,
    dataset_joints=16,
    dataset_channel=list(range(16)),
    inference_channel=list(range(16)))
crowdpose_channel_cfg = dict(
    num_output_channels=14,
    dataset_joints=14,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    ],
    inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
ap10k_channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])
cocowholebody_channel_cfg = dict(
    num_output_channels=133,
    dataset_joints=133,
    dataset_channel=[
        list(range(133)),
    ],
    inference_channel=list(range(133)))

# proto config
num_dataset_keypoints = [channel_cfg['num_output_channels'],
                        aic_channel_cfg['num_output_channels'],
                        mpii_channel_cfg['num_output_channels'],
                        ap10k_channel_cfg['num_output_channels'],
                        ap10k_channel_cfg['num_output_channels'],
                        cocowholebody_channel_cfg['num_output_channels']]
num_in_class_proto = 3

# custom hooks
custom_hooks =[
    dict(
        type='ProtoFreezeHook',
        freeze_epoch=50
    ),
    dict(
        type='MultiheadFreezeHook',
        head_freeze=['freeze,0','thaw,50']
    ),
    dict(
        type='BackboneFreezeHook',
        freeze=['freeze,0', 'thaw,90']
    ),
    dict(
        type='CPSInitHook',
        init_epoch=50,
        cluster_options=[64, 96, 128],
        cluster_iter=500,
        beta=20,
    ),
    dict(
        type='CSSEnableHook',
        start_epoch=50
    )
]

# model settings
model = dict(
    type='TopDownMoEProto',
    pretrained=None,
    backbone=dict(
        type='ViTMoE',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.55,
        num_expert=6,
        part_features=320
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=1280,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1, ),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    associate_keypoint_head=[
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=1280,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1, ),
            out_channels=aic_channel_cfg['num_output_channels'],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=1280,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1, ),
            out_channels=mpii_channel_cfg['num_output_channels'],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=1280,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1, ),
            out_channels=ap10k_channel_cfg['num_output_channels'],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=1280,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1, ),
            out_channels=ap10k_channel_cfg['num_output_channels'],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=1280,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1, ),
            out_channels=cocowholebody_channel_cfg['num_output_channels'],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        ],
    proto_head=dict(
        type='KptProtoHead',
        in_channels=1280,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(
            channels=64,
            head_kernel=3,
            activation='silu',
            neck_type='res',
            num_keypoints=num_dataset_keypoints,
            num_in_class_proto=num_in_class_proto,
            gamma=0.999,
            loss_proto=dict(type='PixelPrototypeCELoss',
                            cfg=dict(weight=1.25e-5,
                                    ppc_weight=0.01,
                                    ppd_weight=0.001,
                                    num_in_class_proto=num_in_class_proto,
                                    num_joints=214)),
            phm_loss_weight=3.33e-6,
            cps_weight=1.0e-2,
            cp_size=96,
            css_weight=1.0e-3,
            css_conf_thr=0.25,
            css_update_weight=3.13e-9,
            css_match_dist=2.1,),
        out_channels=64,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        target_type=target_type,
        modulate_kernel=11,
        use_udp=True))

data_base_dir = '.'

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
    use_gt_bbox=False,
    det_bbox_thr=0.0,
    bbox_file=f'{data_base_dir}/data/coco/person_detection_results/'
    'COCO_val2017_detections_AP_H_56_person.json',
    max_num_joints=133,
    dataset_idx=0,
)

aic_data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=aic_channel_cfg['num_output_channels'],
    num_joints=aic_channel_cfg['dataset_joints'],
    dataset_channel=aic_channel_cfg['dataset_channel'],
    inference_channel=aic_channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='data/coco/person_detection_results/'
    'COCO_val2017_detections_AP_H_56_person.json',
    max_num_joints=133,
    dataset_idx=1,
)

mpii_data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=mpii_channel_cfg['num_output_channels'],
    num_joints=mpii_channel_cfg['dataset_joints'],
    dataset_channel=mpii_channel_cfg['dataset_channel'],
    inference_channel=mpii_channel_cfg['inference_channel'],
    max_num_joints=133,
    dataset_idx=2,
    use_gt_bbox=True,
    bbox_file=None,
)

ap10k_data_cfg = dict(
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
    bbox_file='',
    max_num_joints=133,
    dataset_idx=3,
)

ap36k_data_cfg = dict(
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
    bbox_file='',
    max_num_joints=133,
    dataset_idx=4,
)

cocowholebody_data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=cocowholebody_channel_cfg['num_output_channels'],
    num_joints=cocowholebody_channel_cfg['dataset_joints'],
    dataset_channel=cocowholebody_channel_cfg['dataset_channel'],
    inference_channel=cocowholebody_channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.0,
    bbox_file=f'{data_base_dir}/data/coco/person_detection_results/'
    'COCO_val2017_detections_AP_H_56_person.json',
    dataset_idx=5,
    max_num_joints=133,
)

cocowholebody_train_pipeline = [
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

ap10k_train_pipeline = [
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

aic_train_pipeline = [
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

mpii_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        sigma=2,
        encoding='UDP',
        target_type=target_type),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'flip_pairs', 'dataset_idx'
        ]),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        sigma=2,
        encoding='UDP',
        target_type=target_type),
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
    dict(type='TopDownAffine', use_udp=True),
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

data_root = f'{data_base_dir}/data/coco'
aic_data_root = f'{data_base_dir}/data/aic'
mpii_data_root = f'{data_base_dir}/data/mpii'
ap10k_data_root = f'{data_base_dir}/data/ap10k'
ap36k_data_root = f'{data_base_dir}/data/ap36k'

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=64),
    train=[
        dict(
            type='TopDownCocoDataset',
            ann_file=f'{data_root}/annotations/person_keypoints_train2017.json',
            img_prefix=f'{data_root}/train2017/',
            data_cfg=data_cfg,
            pipeline=train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='TopDownAicDataset',
            ann_file=f'{aic_data_root}/annotations/person_keypoints_train2017.json',
            img_prefix=f'{aic_data_root}/ai_challenger_keypoint_train_20170902/'
            'keypoint_train_images_20170902/',
            data_cfg=aic_data_cfg,
            pipeline=aic_train_pipeline,
            dataset_info={{_base_.aic_info}}),
        dict(
            type='TopDownMpiiDataset',
            ann_file=f'{mpii_data_root}/annotations/mpii_train.json',
            img_prefix=f'{mpii_data_root}/images/',
            data_cfg=mpii_data_cfg,
            pipeline=mpii_train_pipeline,
            dataset_info={{_base_.mpii_info}}),
        dict(
            type='AnimalAP10KDataset',
            ann_file=f'{ap10k_data_root}/annotations/ap10k-train-split1.json',
            img_prefix=f'{ap10k_data_root}/data/',
            data_cfg=ap10k_data_cfg,
            pipeline=ap10k_train_pipeline,
            dataset_info={{_base_.ap10k_info}}),
        dict(
            type='AnimalAP10KDataset',
            ann_file=f'{ap36k_data_root}/annotations/train_annotations_1.json',
            img_prefix=f'{ap36k_data_root}/images/',
            data_cfg=ap36k_data_cfg,
            pipeline=ap10k_train_pipeline,
            dataset_info={{_base_.ap10k_info}}),
        dict(
            type='TopDownCocoWholeBodyDataset',
            ann_file=f'{data_root}/annotations/coco_wholebody_train_v1.0.json',
            img_prefix=f'{data_root}/train2017/',
            data_cfg=cocowholebody_data_cfg,
            pipeline=cocowholebody_train_pipeline,
            dataset_info={{_base_.cocowholebody_info}}),
        ],
    val=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)

