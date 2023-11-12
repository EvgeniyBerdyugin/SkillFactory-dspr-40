_base_ = 'faster-rcnn_r50_fpn_mstrain_3x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20)))

data_root = 'data/voc_coco/'

# train_dataset = dict(
#     dataset=dict(
#         data_root=data_root,
#         ann_file='train.json',
#         data_prefix=dict(img='train/')))

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='val/')))
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'val.json')
test_evaluator = val_evaluator

# training settings
max_epochs = 20 #300
num_last_epochs = 3 #15
interval = 1 #10

default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ),
    early_stopping=dict(
        type="EarlyStoppingHook",
        monitor="coco/bbox_mAP",
        patience=10,
        min_delta=0.005))


load_from = 'checkpoints/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'