_base_ = './../MMDet/mmdetection/configs/centernet/centernet-update_r50-caffe_fpn_ms-1x_coco.py'
#CHANGE#BELOW#
data_root = '/home/sajjad/HW7Leslie/'
train_ann_file = 'train/Results/Detection/train.detection.GTHW7FG.COCO.json'
train_data_prefix = 'train/'
valid_ann_file = 'valid/Results/Detection/valid.detection.GTHW7FG.COCO.json'
valid_data_prefix = 'valid/'
BatchSize = 2
NumWorkers = 2
Epochs = 50
ValInterval = 1
num_classes = 16
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15')
#CHANGE#ABOVE#
load_from = "https://download.openmmlab.com/mmdetection/v3.0/centernet/centernet-update_r50-caffe_fpn_ms-1x_coco/centernet-update_r50-caffe_fpn_ms-1x_coco_20230512_203845-8306baf2.pth"
find_unused_parameters=True # to support backbone freezing

# comming from centernet_update_r50_caffe_fpn_ms_1x_coco
model = dict(
    type='CenterNet',
    # use caffe img_norm
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=4,  # change this number to unfreeze backbone
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        # There is a chance to get 40.3 after switching init_cfg,
        # otherwise it is about 39.9~40.1
        init_cfg=dict(type='Caffe2Xavier', layer='Conv2d'),
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='CenterNetUpdateHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        hm_min_radius=4,
        hm_min_overlap=0.8,
        more_pos_thresh=0.2,
        more_pos_topk=9,
        soft_weight_on_reg=False,
        loss_cls=dict(
            type='GaussianFocalLoss',
            pos_weight=0.25,
            neg_weight=0.75,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
    ),
    train_cfg=None,
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))


# coming from coco detections
# dataset settings
dataset_type = 'CocoDataset'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=BatchSize,
    num_workers=NumWorkers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file= train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=BatchSize,
    num_workers=NumWorkers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file= valid_ann_file,
        data_prefix=dict(img=valid_data_prefix),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + valid_ann_file,
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# comming from schedule 1x
# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=Epochs, val_interval=ValInterval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
base_lr = 0.01
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=1e-3,
        nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        save_best="coco/bbox_mAP",
        rule="greater",
        interval=ValInterval,
        max_keep_ckpts=3
    )
)

# learning rate
param_scheduler = [
    # Use a linear warm-up at [0, 1) epoch
    dict(type='LinearLR',
         start_factor=0.001,
         by_epoch=True,
         begin=0,
         end=1,
         convert_to_iter_based=True
        ),
    dict(
        # use cosine annealing lr from [1, end] epochs
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.01,
        begin=1,
        T_max=Epochs,
        end=Epochs,
        by_epoch=True,
        convert_to_iter_based=True)
]