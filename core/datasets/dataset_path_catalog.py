import os
import os.path as op
from .Cityscapes import CityscapesDataSet
from .vocsdb import vocsbdDataSet
# from .voc import vocDataSet
from .VOC12 import VOC12DataSet
from .gtav import GTAVDataSet
from .synthia import synthiaDataSet
# from .coco_voctrainid import COCODataSetVOCTrainId
from .coco import COCODataSet
import numpy as np
import torch
from torch.utils import data
from PIL import Image
from tqdm import tqdm
import errno
# from utils import resize_img
# from torchvision.transforms import functional as F
from tools.utils import resize_cv_img

class DatasetCatalog(object):
    DATASET_DIR = "datasets"
    VOC_NUM_CLASSES = 21
    CITYSCAPES_NUM_CLASSES = 19
    DATASETS = {
        "gtav_train": {
            "data_dir": "gtav",
            "data_list": "gtav_train_list.txt"
        },
        "synthia_train": {
            "data_dir": "synthia",
            "data_list": "synthia_train_list.txt"
        },
        "cityscapes_train": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_train_list.txt"
        },
        "cityscapes_val": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_val_list.txt"
        },
        "vocsbd_train": {
            "data_dir": "VOCdevkit/VOC2012",
            "data_list": "vocsbd_train_list.txt"
        },
        "voc_train": {
            "data_dir": "VOCdevkit/VOC2012",
            "data_list": "voc_train_list.txt"
        },
        "voc_val": {
            "data_dir": "VOCdevkit/VOC2012",
            "data_list": "voc_val_list.txt"
        },
        "coco_train": {
            "data_dir": "coco",
            "data_list": "coco_train_list.txt"
        },
    }

    @staticmethod
    def get(name, mode, num_classes, max_iters=None, transform=None, cfg=None, empty=False):
        if cfg.DEBUG == 1:
            DatasetCatalog.DATASETS = getDATASETS()
            # DATASETS = getDATASETS()

        if "gtav" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return GTAVDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                               split=mode, transform=transform, debug=cfg.DEBUG)
        elif "synthia" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return synthiaDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                  split=mode, transform=transform, debug=cfg.DEBUG)

        elif "cityscapes" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            if 'coco' in cfg.DATASETS.SOURCE_TRAIN:
                source_dataset = 'coco'
                # max_iters = None
            else:
                source_dataset = 'gtav'
            return CityscapesDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform, cfg=cfg, empty=empty, debug=cfg.DEBUG, source_dataset=source_dataset)

        elif "vocsbd" in name:
            # data_dir = DatasetCatalog.DATASET_DIR
            # attrs = DatasetCatalog.DATASETS[name]
            # args = dict(
            #     root=os.path.join(data_dir, attrs["data_dir"]),
            #     data_list=os.path.join(data_dir, attrs["data_list"]),
            # )
            # return vocsbdDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
            #                          split=mode, transform=transform, cfg=cfg, empty=empty, debug=cfg.DEBUG)

            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return vocsbdDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                               split=mode, transform=transform, debug=cfg.DEBUG)
        elif "voc" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            if mode == 'val':
                num_classes = DatasetCatalog.VOC_NUM_CLASSES
            return VOC12DataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform, cfg=cfg, empty=empty, debug=cfg.DEBUG)

        elif "coco" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            
            # if cfg.DEBUG == 1:
            #     return COCODataSet(args["root"], args["data_list"], max_iters=None, num_classes=num_classes,
            #                    split=mode, transform=transform, debug=cfg.DEBUG)
            if 'cityscapes' in cfg.DATASETS.TARGET_TRAIN:
                max_iters = None
                
            return COCODataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                               split=mode, transform=transform, debug=cfg.DEBUG)
        
        raise RuntimeError("Dataset not available: {}".format(name))

    @staticmethod
    def initMask(cfg):
        if cfg.DEBUG == 1:
            # for i in range(10):
            #     print("Debug without mask initialization!")
            DatasetCatalog.DATASETS = getDATASETS()
            # return
            pass
        # if cfg.ACTIVE.SETTING == 'FS' and cfg.DATASETS.TARGET_TRAIN == "voc_train":
        if cfg.DATASETS.TARGET_TRAIN == "voc_train":
            pass
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS['voc_train']
            data_list = os.path.join(data_dir, attrs["data_list"])
            root = os.path.join(data_dir, attrs["data_dir"])
            with open(data_list, "r") as handle:
                content = handle.readlines()
            idx = 0
            for fname in tqdm(content):
                
                name = fname.strip()
                path2image = os.path.join(root, "JPEGImages/%s.jpg" % (name))
                path2gt = os.path.join(root, "SegmentationClass/%s.png" % (name))
                path2mask = os.path.join(
                    cfg.OUTPUT_DIR,
                    "gtMask/%s/%s"
                    % (
                        "train",
                        name
                        + "_gtFine_labelIds.png",
                    ),
                )
                path2indicator = os.path.join(
                    cfg.OUTPUT_DIR,
                    "gtIndicator/%s/%s"
                    % (
                        "train",
                        name
                        + "_indicator.pth",
                    ),
                )
                mask_dir = os.path.join("%s/gtMask/train/" % (cfg.OUTPUT_DIR))
                indicator_dir = os.path.join("%s/gtIndicator/train/" % (cfg.OUTPUT_DIR))

                # mkdir
                if not os.path.exists(mask_dir):
                    mkdir_path(mask_dir)
                if not os.path.exists(indicator_dir):
                    mkdir_path(indicator_dir)

                img = Image.open(path2image).convert('RGB')
                # resize voc image to 512*512
                # img = F.resize(img, (512, 512), Image.BICUBIC)
                img = resize_cv_img(img)
                h, w = img.size[1], img.size[0]
                # mask = np.ones((h, w), dtype=np.uint8) * 255
                
                if idx < cfg.ACTIVE.LABELED:
                    
                    mask = Image.open(path2gt)
                    #****************************************
                    # ignore_label = 255
                    mask = resize_cv_img(mask)
                    mask = np.array(mask, dtype=np.uint8)
                    # mask_copy = np.ones_like(mask, dtype=np.uint8) * ignore_label
                    # mask_copy = ignore_label * np.ones(mask.shape, dtype=np.uint8)
                    # id_to_trainid = {0: 0, 1: 5, 2: 2, 3: 15, 4: 9, 5: 40, 6: 6, 7: 3, 8: 16, 9: 57, 10: 20, 11: 61, 12: 17, 13: 18, 14: 4, 15: 1, 16: 59, 17: 19, 18: 58, 19: 7, 20: 63}
                
                    # for k, v in id_to_trainid.items():
                    #     mask_copy[mask == k] = v
                    # mask = np.array(mask_copy, dtype=np.uint8)
                    #******************************************
                    
                    mask = Image.fromarray(mask)
                    # resize mask to 512*512
                    # mask = F.resize(mask, (512, 512), Image.BICUBIC)
                    # mask = resize_cv_img(mask)
                else:
                    mask = np.ones((h, w), dtype=np.uint8) * 255
                    mask = Image.fromarray(mask)
                    # resize mask to 512*512
                    # mask = F.resize(mask, (512, 512), Image.BICUBIC)
                    mask = resize_cv_img(mask)
                mask.save(path2mask)

                indicator = {
                    'active': torch.tensor([0], dtype=torch.bool),
                    'selected': torch.tensor([0], dtype=torch.bool),
                }
                torch.save(indicator, path2indicator)

                idx += 1 
        elif cfg.DATASETS.TARGET_TRAIN == "cityscapes_train":
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS['cityscapes_train']
            data_list = os.path.join(data_dir, attrs["data_list"])
            root = os.path.join(data_dir, attrs["data_dir"])
            with open(data_list, "r") as handle:
                content = handle.readlines()
            idx = 0
            for fname in tqdm(content):
                name = fname.strip()
                path2image = os.path.join(root, "leftImg8bit/%s/%s" % ('train', name))
                path2gt = os.path.join(root, "segmentation/%s/%s" % ('train', os.path.basename(name).replace("leftImg8bit", "gtFine")))
                path2mask = os.path.join(
                    cfg.OUTPUT_DIR,
                    "gtMask/%s/%s"
                    % (
                        "train",
                        name.split("_leftImg8bit")[0]
                        + "_gtFine_labelIds.png",
                    ),
                )
                path2indicator = os.path.join(
                    cfg.OUTPUT_DIR,
                    "gtIndicator/%s/%s"
                    % (
                        "train",
                        name.split("_leftImg8bit")[0]
                        + "_indicator.pth",
                    ),
                )
                mask_dir = os.path.join("%s/gtMask/train/%s" % (cfg.OUTPUT_DIR, name.split("/")[0]))
                indicator_dir = os.path.join("%s/gtIndicator/train/%s" % (cfg.OUTPUT_DIR, name.split("/")[0]))

                # mkdir
                if not os.path.exists(mask_dir):
                    mkdir_path(mask_dir)
                if not os.path.exists(indicator_dir):
                    mkdir_path(indicator_dir)

                img = Image.open(path2image).convert('RGB')
                h, w = img.size[1], img.size[0]
                # mask = np.ones((h, w), dtype=np.uint8) * 255
                # mask = Image.fromarray(mask)
                if idx < cfg.ACTIVE.LABELED:
                    mask = Image.open(path2gt)
                    #****************************************
                    # ignore_label = 255
                    mask = resize_cv_img(mask, mask.size[1], mask.size[0])
                    mask = np.array(mask, dtype=np.uint8)
                    mask = Image.fromarray(mask)
                else:
                    mask = np.ones((h, w), dtype=np.uint8) * 255
                    mask = Image.fromarray(mask)
                    # resize mask to 512*512
                    # mask = F.resize(mask, (512, 512), Image.BICUBIC)
                    mask = resize_cv_img(mask, mask.size[1], mask.size[0])
                mask.save(path2mask)

                indicator = {
                    'active': torch.tensor([0], dtype=torch.bool),
                    'selected': torch.tensor([0], dtype=torch.bool),
                }
                torch.save(indicator, path2indicator)
                idx += 1
        

def mkdir_path(dir):
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def getDATASETS():
    return {
                "gtav_train": {
                    "data_dir": "gtav",
                    "data_list": "gtav_train_list_subset.txt"
                },
                "coco_train": {
                    "data_dir": "coco",
                    "data_list": "coco_train_list_subset.txt"
                },
                "synthia_train": {
                    "data_dir": "synthia",
                    "data_list": "synthia_train_list_subset.txt"
                },
                "cityscapes_train": {
                    "data_dir": "cityscapes",
                    "data_list": "cityscapes_train_list_subset.txt"
                },
                "cityscapes_val": {
                    "data_dir": "cityscapes",
                    "data_list": "cityscapes_val_list_subset.txt"
                },
                "cityscapes_train_debug": {
                    "data_dir": "cityscapes_debug",
                    "data_list": "cityscapes_train_list_subset.txt"
                },
                "cityscapes_val_debug": {
                    "data_dir": "cityscapes_debug",
                    "data_list": "cityscapes_val_list_subset.txt"
                },
                "vocsbd_train": {
                    "data_dir": "VOCdevkit/VOC2012",
                    "data_list": "vocsbd_train_list_subset.txt"
                },
                # "voc_train": {
                #     "data_dir": "VOCdevkit/VOC2012",
                #     "data_list": "voc_train_list_subset.txt"
                # },
                # "voc_val": {
                #     "data_dir": "VOCdevkit/VOC2012",
                #     "data_list": "voc_val_list_subset.txt"
                # },
                
                "voc_train": {
                    "data_dir": "VOCdevkit/VOC2012",
                    "data_list": "voc_train_list.txt"
                },
                "voc_val": {
                    "data_dir": "VOCdevkit/VOC2012",
                    "data_list": "voc_val_list.txt"
                },
            }
