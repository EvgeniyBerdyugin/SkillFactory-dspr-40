
# Inherit and overwrite part of the config based on this config
_base_ = './yolox_s_8xb8-300e_coco.py'

data_root = 'data/voc_coco/' # dataset root

# train_batch_size_per_gpu = 4
# train_num_workers = 2

max_epochs = 20


# metainfo = {
#     'classes': ('person',
#                 'bird',
#                 'chair',
#                 'pottedplant',
#                 'sofa',
#                 'cat',
#                 'boat',
#                 'train',
#                 'car',
#                 'aeroplane',
#                 'dog',
#                 'sheep',
#                 'horse',
#                 'diningtable',
#                 'bicycle',
#                 'bus',
#                 'tvmonitor',
#                 'bottle',
#                 'motorbike',
#                 'cow'
#     ),
#     'palette': [(220, 20, 60),
#                 (220, 220, 60),
#                 (220, 20, 255),
#                 (220, 20, 60),
#                 (255, 20, 60),
#                 (255, 220, 60),
#                 (255, 20, 255),
#                 (220, 255, 60),
#                 (220, 255, 255),
#                 (100, 20, 60),
#                 (100, 220, 60),
#                 (100, 20, 255),
#                 (100, 100, 60),
#                 (100, 20, 100),
#                 (0, 20, 60),
#                 (0, 220, 60),
#                 (0, 20, 255),
#                 (0, 220, 255),
#                 (0, 220, 0),
#                 (150, 150, 150),
#     ]
# }


train_dataset = dict(
    dataset=dict(
        data_root=data_root,
        # metainfo=metainfo,
        data_prefix=dict(img='train/'),
        ann_file='train.json'))


train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        # metainfo=metainfo,
        data_prefix=dict(img='train/'),
        ann_file='train.json'))

        
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        # metainfo=metainfo,
        data_prefix=dict(img='val/'),
        ann_file='val.json'))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'val.json')

test_evaluator = val_evaluator

model = dict(bbox_head=dict(num_classes=20))


# load COCO pre-trained weight
# load_from = './checkpoints/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'

# visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])
