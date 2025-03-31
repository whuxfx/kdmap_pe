# model settings
model = dict(
    type='SMOKEMono3D',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='DLANet',
        depth=34,
        in_channels=3,
        norm_cfg=dict(type='GN', num_groups=32),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth'
        )),
    neck=dict(
        type='DLANeck',
        in_channels=[16, 32, 64, 128, 256, 512],
        start_level=2,
        end_level=5,
        norm_cfg=dict(type='GN', num_groups=32)),
    bbox_head=dict(
        type='SMOKEMono3DHead',
        num_classes=3,
        in_channels=64,
        dim_channel=[3, 4, 5],
        ori_channel=[6, 7],
        stacked_convs=0,
        feat_channels=64,
        use_direction_classifier=False,
        diff_rad_by_sin=False,
        pred_attrs=False,
        pred_velo=False,
        dir_offset=0,
        strides=None,
        group_reg_dims=(8, ),
        cls_branch=(256, ),
        reg_branch=((256, ), ),
        num_attrs=0,
        bbox_code_size=7,
        dir_branch=(),
        attr_branch=(),
        bbox_coder=dict(
            type='SMOKECoder',
            base_depth=(28.01, 16.32),
            base_dims=((0.88, 1.73, 0.67), (1.78, 1.70, 0.58), (3.88, 1.63,
                                                                1.53)),
            code_size=7),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='sum', loss_weight=1 / 300),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_attr=None,
        conv_bias=True,
        dcn_on_last_conv=False),
    train_cfg=None,
    test_cfg=dict(topK=100, local_maximum_kernel=3, max_per_img=100))
