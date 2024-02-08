_base_ = './../MMDet/mmdetection/configs/yolox/yolox_x_8xb8-300e_coco.py'
#CHANGE#BELOW#
data_root = '/home/sajjad/HW7Leslie/'
train_ann_file = 'train/Results/Detection/train.detection.GTHW7.COCO.json'
train_data_prefix = 'train/'
valid_ann_file = 'valid/Results/Detection/valid.detection.GTHW7.COCO.json'
valid_data_prefix = 'valid/'
BatchSize = 2
NumWorkers = 2
Epochs = 20
ValInterval = 1
num_classes = 80
classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
#CHANGE#ABOVE#
load_from = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"
find_unused_parameters=True # to support backbone freezing
model = dict(
    backbone=dict(deepen_factor=1.33, widen_factor=1.25, frozen_stages=4),
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
# test_cfg=dict(score_thr=0.25, nms=dict(type='nms', iou_threshold=0.65))

# optimizer
# default 8 gpu
base_lr = 0.01
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=1e-3,
        nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

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

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        save_best="coco/bbox_mAP",
        rule="greater",
        interval=ValInterval,
        max_keep_ckpts=3
    )
)