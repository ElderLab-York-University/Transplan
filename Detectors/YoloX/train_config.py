_base_ = './../MMDet/mmdetection/configs/yolox/yolox_x_8xb8-300e_coco.py'
#CHANGE#BELOW#
data_root = '/home/sajjad/Transplan/DemoDataSet/'
train_ann_file = 'Split1/Results/Detection/Split1.detection.GTHW7.COCO.json'
train_data_prefix = 'Split1/'
valid_ann_file = 'Split1/Results/Detection/Split1.detection.GTHW7.COCO.json'
valid_data_prefix = 'Split1/'
BatchSize = 6
NumWorkers = 8
Epochs = 100
ValInterval = 10
num_classes = 4
classes = ('0', '2', '5', '7')
#CHANGE#ABOVE#
load_from = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"
model = dict(
    backbone=dict(deepen_factor=1.33, widen_factor=1.25),
    neck=dict(
        in_channels=[320, 640, 1280], out_channels=320, num_csp_blocks=4),
    bbox_head=dict(in_channels=320, feat_channels=320, num_classes=num_classes))

train_dataset = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file=train_ann_file ,
        data_prefix=dict(img=train_data_prefix)
        )
    )

train_dataloader = dict(
    batch_size=BatchSize,
    num_workers=NumWorkers,
    dataset=train_dataset
    )

val_dataloader = dict(
    batch_size=BatchSize,
    num_workers=NumWorkers,
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file=valid_ann_file,
        data_prefix=dict(img=valid_data_prefix)
        )
    )

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + valid_ann_file,
    metric=['bbox']
    )

test_dataloader = val_dataloader
test_evaluator = val_evaluator

train_cfg = dict(max_epochs=Epochs, val_interval=ValInterval)

# optimizer
# default 8 gpu
base_lr = 0.01
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4,
        nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

param_scheduler = [
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=0,
        T_max=Epochs,
        end=Epochs,
        by_epoch=True,
        convert_to_iter_based=True)
]

default_hooks = dict(
    checkpoint=dict(
        interval=ValInterval,
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ))

custom_hooks = []