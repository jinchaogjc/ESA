import argparse
import os
import datetime
import logging
import time
import math
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import re

import torch
import torch.nn.functional as F
import torch.backends.cudnn

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_feature_extractor, build_classifier
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU, get_color_pallete
from core.utils.logger import setup_logger

from core.datasets.build import build_transform
from core.datasets.dataset_path_catalog import DatasetCatalog
from core.active.floating_region import FloatingRegionScore
from core.active.spatial_purity import SpatialPurity

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def inference(feature_extractor, classifier, image, label, flip=False):
    size = label.shape[-2:]
    if flip:
        image = torch.cat([image, torch.flip(image, [3])], 0)
        image = image.cuda()
    with torch.no_grad():
        output = classifier(feature_extractor(image))
    output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    return output.unsqueeze(dim=0)


def transform_color(pred):
    synthia_to_city = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 10,
        10: 11,
        11: 12,
        12: 13,
        13: 15,
        14: 17,
        15: 18,
    }
    label_copy = 255 * np.ones(pred.shape, dtype=np.float32)
    for k, v in synthia_to_city.items():
        label_copy[pred == k] = v
    return label_copy.copy()

# def transform_color_cocovoc(pred):
    coco_to_voc = {  
                    0:  0,
                    5:  1,
                    2:  2,
                    15: 3,
                    9:  4,
                    40: 5,
                    6:  6,
                    3:  7,
                    16: 8,
                    57: 9,
                    20: 10,
                    61: 11,
                    17: 12,
                    18: 13,
                    4:  14,
                    1:  15,
                    59: 16,
                    19: 17,
                    58: 18,
                    7:  19,
                    63: 20}
    label_copy = 255 * np.ones(pred.shape, dtype=np.float32)
    for k, v in coco_to_voc.items():
        label_copy[pred == k] = v
    return label_copy.copy()



# def transform_color_cococityscapes(pred):
    coco_to_cityscapes = {82: 0, 83: 1, 84: 2, 85: 3, 86: 4, 87: 5, 10: 6, 12: 7, 88: 8, 89: 9, 90: 10, 1: 11, 91: 12, 3: 13, 8: 14, 6: 15, 7: 16, 4: 17, 2: 18}
    label_copy = 255 * np.ones(pred.shape, dtype=np.float32)
    for k, v in coco_to_cityscapes.items():
        label_copy[pred == k] = v
    return label_copy.copy()


# def get_coco_trainid(num_classes):
    id_to_trainid = {  
                     0:  0,
                    5:  1,
                    2:  2,
                    15: 3,
                    9:  4,
                    40: 5,
                    6:  6,
                    3:  7,
                    16: 8,
                    57: 9,
                    20: 10,
                    61: 11,
                    17: 12,
                    18: 13,
                    4:  14,
                    1:  15,
                    59: 16,
                    19: 17,
                    58: 18,
                    7:  19,
                    63: 20}
        
    if num_classes == 21:
        trainid2name = {
            0: 'background',
            1: 'aeroplane',
            2: 'bicycle',
            3: 'bird',
            4: 'boat',
            5: 'bottle',
            6: 'bus',
            7: 'car',
            8: 'cat',
            9: 'chair',
            10: 'cow',
            11: 'diningtable',
            12: 'dog',
            13: 'horse',
            14: 'motorbike',
            15: 'person',
            16: 'pottedplant',
            17: 'sheep',
            18: 'sofa',
            19: 'train',
            20: 'tvmonitor',
        }
    elif num_classes == 19:
        trainid2name = {
                0: "road",
                1: "sidewalk",
                2: "building",
                3: "wall",
                4: "fence",
                5: "pole",
                6: "light",
                7: "sign",
                8: "vegetation",
                9: "terrain",
                10: "sky",
                11: "person",
                12: "rider",
                13: "car",
                14: "truck",
                15: "bus",
                16: "train",
                17: "motocycle",
                18: "bicycle",
            }
    else:
    
    # CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    #            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    #            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    #            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    #            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    #            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    #            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    #            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    #            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    #            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    #            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    #            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    #            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    #            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    
        trainid2name = {0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 12: 'stop sign', 13: 'parking meter', 14: 'bench', 15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe', 25: 'backpack', 26: 'umbrella', 27: 'handbag', 28: 'tie', 29: 'suitcase', 30: 'frisbee', 31: 'skis', 32: 'snowboard', 33: 'sports ball', 34: 'kite', 35: 'baseball bat', 36: 'baseball glove', 37: 'skateboard', 38: 'surfboard', 39: 'tennis racket', 40: 'bottle', 41: 'wine glass', 42: 'cup', 43: 'fork', 44: 'knife', 45: 'spoon', 46: 'bowl', 47: 'banana', 48: 'apple', 49: 'sandwich', 50: 'orange', 51: 'broccoli', 52: 'carrot', 53: 'hot dog', 54: 'pizza', 55: 'donut', 56: 'cake', 57: 'chair', 58: 'couch', 59: 'potted plant', 60: 'bed', 61: 'dining table', 62: 'toilet', 63: 'tv', 64: 'laptop', 65: 'mouse', 66: 'remote', 67: 'keyboard', 68: 'cell phone', 69: 'microwave', 70: 'oven', 71: 'toaster', 72: 'sink', 73: 'refrigerator', 74: 'book', 75: 'clock', 76: 'vase', 77: 'scissors', 78: 'teddy bear', 79: 'hair drier', 80: 'toothbrush'}
    return trainid2name

def test(cfg, mode="test"):
    
    # if cfg.DATASETS.SOURCE_TRAIN == "coco_train":
    #     # cfg.MODEL.NUM_CLASSES = DatasetCatalog.VOC_NUM_CLASSES
    #     NUM_CLASSES = DatasetCatalog.VOC_NUM_CLASSES
    # else:
    #     NUM_CLASSES = cfg.MODEL.NUM_CLASSES
    NUM_CLASSES = cfg.MODEL.NUM_CLASSES
    
    logger = logging.getLogger("AL-RIPU.tester")
    logger.info("Start testing")
    device = torch.device(cfg.MODEL.DEVICE)

    feature_extractor = build_feature_extractor(cfg)
    feature_extractor.to(device)

    classifier = build_classifier(cfg)
    classifier.to(device)

    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        feature_extractor_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(feature_extractor_weights)
        classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)

    feature_extractor.eval()
    classifier.eval()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    torch.cuda.empty_cache()
    dataset_name = cfg.DATASETS.TEST
    output_folder = '.'
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        if not os.path.exists(output_folder):
            mkdir(output_folder)


    # if mode == "train":
    #     transform = build_transform(cfg, mode, is_source=False)
    #     test_data = DatasetCatalog.get(cfg.DATASETS.TARGET_TRAIN, mode, num_classes=NUM_CLASSES,
    #                        max_iters=None, transform=transform, cfg=cfg, empty=False)

    # elif mode == "select":
    #     mode = "train"
    #     iters = cfg.SOLVER.MAX_ITER * cfg.SOLVER.BATCH_SIZE
    #     transform = build_transform(cfg, mode, is_source=False)
    #     test_data = DatasetCatalog.get(cfg.DATASETS.TARGET_TRAIN, mode, num_classes=NUM_CLASSES,
    #                                    max_iters=iters, transform=transform, cfg=cfg, empty=False)
    # else:
    mode = "test"
    test_data = build_dataset(cfg, mode=mode, is_source=False)

    assert cfg.TEST.BATCH_SIZE == 1, "Test batch size should be 1!"
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4,
                                              pin_memory=True, sampler=None)
    score_dict = {}
    for batch in tqdm(test_loader):
        x, y, name = batch['img'], batch['label'], batch['name']
        name = name[0]
        # print(name)
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True).long()
        # pdb.set_trace()
        pred = inference(feature_extractor, classifier, x, y, True)
        # if cfg.DEBUG == 2:
        #     floating_region_score = FloatingRegionScore(in_channels=NUM_CLASSES,
        #                                                 size=2 * cfg.ACTIVE.RADIUS_K + 1).cuda()
        #     score, purity, entropy = floating_region_score(pred)
            
        #     score_sum = sum(sum(score))
        #     score_avg = score_sum/score.shape[0]/score.shape[1]
        #     print("score sum:", score_sum)
        #     print("score avg:", score_avg)
        #     score_dict[name] = [score_sum.item(), score_avg.item()]
            
        # if cfg.DEBUG:
        #     # pdb.set_trace()
        #     calculate_purity = SpatialPurity(in_channels=cfg.MODEL.NUM_CLASSES, size=2 * cfg.ACTIVE.RADIUS_K + 1).cuda()
        #     p = torch.softmax(pred[0], dim=0)
        #     entropy = torch.sum(-p * torch.log(p + 1e-6), dim=0)
        #     pseudo_label = torch.argmax(p, dim=0)
        #     one_hot = F.one_hot(pseudo_label, num_classes=cfg.MODEL.NUM_CLASSES).float()
        #     one_hot = one_hot.permute((2, 0, 1)).unsqueeze(dim=0)
        #     purity = calculate_purity(one_hot).squeeze(dim=0).squeeze(dim=0)
        #     score = entropy * purity

        # if cfg.DEBUG:
        #     from core.active.build import PixelSelection
        #     PixelSelection(cfg, feature_extractor, classifier, test_loader)
        # pdb.set_trace()
        
        
            
        output = pred.max(1)[1]
        # if cfg.DATASETS.SOURCE_TRAIN == "coco_train":
        #     # change output from 81 labels to 21 labels
        #     pass
        intersection, union, target = intersectionAndUnionGPU(output, y, NUM_CLASSES, cfg.INPUT.IGNORE_LABEL)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

        # save the result
        pred = pred.cpu().numpy().squeeze().argmax(0)
        # if NUM_CLASSES == 16:
        pred = transform_color(pred)
        
        # if "cityscapes" in cfg.DATASETS.TARGET_TRAIN:
        #     color_name = "city"
        #     mask_filename = name if len(name.split("/")) < 2 else name.split("/")[1]
        # elif "voc" in cfg.DATASETS.TARGET_TRAIN:
        #     color_name = "voc"
        #     mask_filename = name + ".png"
        # else:
        #     color_name = "NOT DEFINED"
            
        # if "cityscapes" in cfg.DATASETS.TARGET_TRAIN and "coco" in cfg.DATASETS.SOURCE_TRAIN:
        #     color_name = "city"
        #     mask_filename = name if len(name.split("/")) < 2 else name.split("/")[1]
        #     pred = transform_color_cococityscapes(pred)
            
        # if "voc" in cfg.DATASETS.TARGET_TRAIN and "coco" in cfg.DATASETS.SOURCE_TRAIN:
        #     color_name = "voc"
        #     mask_filename = name + ".png"
        #     pred = transform_color_cocovoc(pred)
            
        # # mask = get_color_pallete(pred, "city")
        if "cityscapes" in cfg.DATASETS.TARGET_TRAIN:
            color_name = "city"
            mask_filename = os.path.basename(name)
        elif "voc" in cfg.DATASETS.TARGET_TRAIN:
            color_name = "voc"
            mask_filename = name + ".png"
        mask = get_color_pallete(pred, color_name)
        
        if mask.mode == 'P':
            mask = mask.convert('RGB')
        mask.save(os.path.join(output_folder, mask_filename))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    # if cfg.DATASETS.SOURCE_TRAIN == "coco_train" and cfg.DATASETS.TARGET_TRAIN == "voc_train":
    #     TARGET_NUM_CLASSES = DatasetCatalog.VOC_NUM_CLASSES
    #     voc_to_cocoid = {0: 0, 1: 5, 2: 2, 3: 15, 4: 9, 5: 40, 6: 6, 7: 3, 8: 16, 9: 57, 10: 20, 11: 61, 12: 17, 13: 18, 14: 4, 15: 1, 16: 59, 17: 19, 18: 58, 19: 7, 20: 63}   
    #     target_to_sourceid = voc_to_cocoid
    # if cfg.DATASETS.SOURCE_TRAIN == "coco_train" and cfg.DATASETS.TARGET_TRAIN == "cityscapes_train":
    #     TARGET_NUM_CLASSES = 19
    #     cityscapes_to_cocoid = {0: 82, 1: 83, 2: 84, 3: 85, 4: 86, 5: 87, 6: 10, 7: 12, 8: 88, 9: 89, 10: 90, 11: 1, 12: 91, 13: 3, 14: 8, 15: 6, 16: 7, 17: 4, 18: 2}
    #     target_to_sourceid = cityscapes_to_cocoid
    # # if cfg.DATASETS.SOURCE_TRAIN == "coco_train":
    # # change iou_class from 81 class to 21 class
    # iou_class_tmp = [i for i in range(TARGET_NUM_CLASSES)]
    # # voc_to_cocoid = {0: 0, 1: 5, 2: 2, 3: 15, 4: 9, 5: 40, 6: 6, 7: 3, 8: 16, 9: 57, 10: 20, 11: 61, 12: 17, 13: 18, 14: 4, 15: 1, 16: 59, 17: 19, 18: 58, 19: 7, 20: 63}   
    
    # for i in range(TARGET_NUM_CLASSES):
    #     iou_class_tmp[i] = iou_class[target_to_sourceid[i]]
    # accuracy_class_tmp = [i for i in range(TARGET_NUM_CLASSES)]
    # for i in range(TARGET_NUM_CLASSES):
    #     accuracy_class_tmp[i] = accuracy_class[target_to_sourceid[i]]
    # iou_class = np.array(iou_class_tmp)
    # accuracy_class = np.array(accuracy_class_tmp)
    
        
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)


    # if cfg.DATASETS.SOURCE_TRAIN == "coco_train" and cfg.DATASETS.TARGET_TRAIN == "voc_train":
    #     NUM_CLASSES = DatasetCatalog.VOC_NUM_CLASSES
    #     trainid2name = get_coco_trainid(NUM_CLASSES)
    # elif cfg.DATASETS.SOURCE_TRAIN == "coco_train" and cfg.DATASETS.TARGET_TRAIN == "cityscapes_train":
    #     NUM_CLASSES = DatasetCatalog.CITYSCAPES_NUM_CLASSES
    #     trainid2name = get_coco_trainid(NUM_CLASSES)
    # else:
    #     trainid2name = test_data.trainid2name
    #     NUM_CLASSES = cfg.MODEL.NUM_CLASSES
        
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(NUM_CLASSES):
        logger.info(
            '{} {} iou/accuracy: {:.4f}/{:.4f}.'.format(i, test_data.trainid2name[i], iou_class[i], accuracy_class[i]))

    # if cfg.DEBUG == 2:
    #     print("score_dict:", score_dict)
    #     write_file_name = "score_dict_2975.txt"
    #     write_file = open(write_file_name, "a")
    #     write_file.write(str(score_dict))
    #     write_file.close()
    

def main():
    parser = argparse.ArgumentParser(description="Active Domain Adaptive Semantic Segmentation Testing")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        )
    parser.add_argument("--proctitle",
                        type=str,
                        default="AL-RIPU",
                        help="allow a process to change its title", )
    parser.add_argument("--mode",
                        type=str,
                        default="test",
                        help="input mode: train, test")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("AL-RIPU", save_dir, 0)
    logger.info(cfg)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    test(cfg, args.mode)


if __name__ == "__main__":
    main()
