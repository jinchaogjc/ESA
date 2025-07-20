import math
import torch

import numpy as np
import torch.nn.functional as F


from PIL import Image
from tqdm import tqdm
from .floating_region import FloatingRegionScore
from .spatial_purity import SpatialPurity
import time
# compile snic
import os
from skimage.segmentation import slic


def PixelSelection(cfg, feature_extractor, classifier, tgt_epoch_loader):
    feature_extractor.eval()
    classifier.eval()

    # SELECTED_PIXELS / SELECTED_ITER_TIMES / TARGET_INPUT_SIZE * TARGET_ORIGINAL_SIZE
    # for example:
    # from source_dataset GTAV to Cityscapes
    # 40 pixels / len([10000, 12000, 14000, 16000, 18000]) / (1280 * 640) * (2048 * 1024)
    # GTAV original image size: (1914 * 1052)
    # active_pixels = math.ceil(cfg.ACTIVE.PIXELS / len(cfg.ACTIVE.SELECT_ITER) / (1280 * 640) * (2048 * 1024))
    calculate_purity = SpatialPurity(in_channels=cfg.MODEL.NUM_CLASSES, size=2 * cfg.ACTIVE.RADIUS_K + 1).cuda()
    mask_radius = cfg.ACTIVE.RADIUS_K

    with torch.no_grad():
        # for tgt_data in tqdm(tgt_epoch_loader):
        # **********!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for index, tgt_data in enumerate(tgt_epoch_loader):
            # if the image is labeled, skip it
            # target epoch loader with fixed batch_size=1,
            if index < cfg.ACTIVE.LABELED:
                continue

            tgt_input, path2mask = tgt_data['img'], tgt_data['path_to_mask']
            origin_mask, origin_label = tgt_data['origin_mask'], tgt_data['origin_label']
            origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            tgt_input = tgt_input.cuda(non_blocking=True)

            tgt_size = tgt_input.shape[-2:]
            tgt_feat = feature_extractor(tgt_input)
            tgt_out = classifier(tgt_feat, size=tgt_size)

            for i in range(len(origin_mask)):

                active_mask = origin_mask[i].cuda(non_blocking=True)
                ground_truth = origin_label[i].cuda(non_blocking=True)
                size = (origin_size[i][0], origin_size[i][1])
                active = active_indicator[i]
                selected = selected_indicator[i]

                output = tgt_out[i:i + 1, :, :, :]
                output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
                output = output.squeeze(dim=0)
                p = torch.softmax(output, dim=0)
                entropy = torch.sum(-p * torch.log(p + 1e-6), dim=0)
                pseudo_label = torch.argmax(p, dim=0)
                one_hot = F.one_hot(pseudo_label, num_classes=cfg.MODEL.NUM_CLASSES).float()
                one_hot = one_hot.permute((2, 0, 1)).unsqueeze(dim=0)
                purity = calculate_purity(one_hot).squeeze(dim=0).squeeze(dim=0)
                if cfg.ACTIVE.LABOR == "ENT":
                    score = entropy
                elif cfg.ACTIVE.LABOR == "PUR":
                    score = purity
                else:
                    score = entropy * purity
                

                score[active] = -float('inf')

                # active_pixels = math.ceil(cfg.ACTIVE.PIXELS / len(cfg.ACTIVE.SELECT_ITER) / (cfg.INPUT.TARGET_INPUT_SIZE_TRAIN[0] * cfg.INPUT.TARGET_INPUT_SIZE_TRAIN[1]) * (tgt_input.shape[-2:][0] * tgt_input.shape[-2:][0]))
                active_pixels = math.ceil(cfg.ACTIVE.PIXELS / len(cfg.ACTIVE.SELECT_ITER) / (cfg.INPUT.TARGET_INPUT_SIZE_TRAIN[0] * cfg.INPUT.TARGET_INPUT_SIZE_TRAIN[1]) * (origin_mask.shape[-2:][1] * origin_mask.shape[-2:][0]))
                
                for pixel in range(active_pixels):
                    values, indices_h = torch.max(score, dim=0)
                    _, indices_w = torch.max(values, dim=0)
                    w = indices_w.item()
                    h = indices_h[w].item()

                    start_w = w - mask_radius if w - mask_radius >= 0 else 0
                    start_h = h - mask_radius if h - mask_radius >= 0 else 0
                    end_w = w + mask_radius + 1
                    end_h = h + mask_radius + 1
                    # mask out
                    score[start_h:end_h, start_w:end_w] = -float('inf')
                    active[start_h:end_h, start_w:end_w] = True
                    selected[h, w] = True
                    # active sampling
                    active_mask[h, w] = ground_truth[h, w]

                active_mask = Image.fromarray(np.array(active_mask.cpu().numpy(), dtype=np.uint8))
                active_mask.save(path2mask[i])
                indicator = {
                    'active': active,
                    'selected': selected
                }
                torch.save(indicator, path2indicator[i])

    feature_extractor.train()
    classifier.train()


def RegionSelection(cfg, feature_extractor, classifier, tgt_epoch_loader):

    feature_extractor.eval()
    classifier.eval()

    floating_region_score = FloatingRegionScore(in_channels=cfg.MODEL.NUM_CLASSES, size=2 * cfg.ACTIVE.RADIUS_K + 1).cuda()
    per_region_pixels = (2 * cfg.ACTIVE.RADIUS_K + 1) ** 2
    active_radius = cfg.ACTIVE.RADIUS_K
    mask_radius = cfg.ACTIVE.RADIUS_K * 2
    active_ratio = cfg.ACTIVE.RATIO / len(cfg.ACTIVE.SELECT_ITER)

    with torch.no_grad():
        # for tgt_data in tqdm(tgt_epoch_loader):
        # **********!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for index, tgt_data in enumerate(tgt_epoch_loader):
            # if the image is labeled, skip it
            # target epoch loader with fixed batch_size=1,
            if index < cfg.ACTIVE.LABELED:
                continue

            tgt_input, path2mask = tgt_data['img'], tgt_data['path_to_mask']
            origin_mask, origin_label = \
                tgt_data['origin_mask'], tgt_data['origin_label']
            origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            tgt_input = tgt_input.cuda(non_blocking=True)

            tgt_size = tgt_input.shape[-2:]
            tgt_feat = feature_extractor(tgt_input)
            tgt_out = classifier(tgt_feat, size=tgt_size)

            for i in range(len(origin_mask)):
                active_mask = origin_mask[i].cuda(non_blocking=True)
                ground_truth = origin_label[i].cuda(non_blocking=True)
                size = (origin_size[i][0], origin_size[i][1])
                num_pixel_cur = size[0] * size[1]
                active = active_indicator[i]
                selected = selected_indicator[i]

                output = tgt_out[i:i + 1, :, :, :]
                output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
                score, purity, entropy = floating_region_score(output)

                score[active] = -float('inf')

                active_regions = math.ceil(num_pixel_cur * active_ratio / per_region_pixels)

                for pixel in range(active_regions):
                    values, indices_h = torch.max(score, dim=0)
                    _, indices_w = torch.max(values, dim=0)
                    w = indices_w.item()
                    h = indices_h[w].item()

                    active_start_w = w - active_radius if w - active_radius >= 0 else 0
                    active_start_h = h - active_radius if h - active_radius >= 0 else 0
                    active_end_w = w + active_radius + 1
                    active_end_h = h + active_radius + 1

                    mask_start_w = w - mask_radius if w - mask_radius >= 0 else 0
                    mask_start_h = h - mask_radius if h - mask_radius >= 0 else 0
                    mask_end_w = w + mask_radius + 1
                    mask_end_h = h + mask_radius + 1

                    # mask out
                    score[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = -float('inf')
                    active[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = True
                    selected[active_start_h:active_end_h, active_start_w:active_end_w] = True
                    # active sampling
                    active_mask[active_start_h:active_end_h, active_start_w:active_end_w] = \
                        ground_truth[active_start_h:active_end_h, active_start_w:active_end_w]

                active_mask = Image.fromarray(np.array(active_mask.cpu().numpy(), dtype=np.uint8))
                active_mask.save(path2mask[i])
                indicator = {
                    'active': active,
                    'selected': selected
                }
                torch.save(indicator, path2indicator[i])

    feature_extractor.train()
    classifier.train()


def load_image(imgname, resize=False, w=512, h=512):
    #********************load image********************
    img = Image.open(imgname)
    if resize:
        img = Func.resize(img, (h, w), Image.NEAREST)
    # img = imread(imgname)
    img = np.asarray(img)
    #***************************************************
    return img

def SuperPixelSelection(cfg, tgt_epoch_loader):
  
    with torch.no_grad():
        # tgt_epoch_loader with fixed batch_size=1
        for tgt_data in tqdm(tgt_epoch_loader):

            tgt_input, path2mask, name = tgt_data['img'], tgt_data['path_to_mask'], tgt_data['name']
            origin_mask, origin_label = tgt_data['origin_mask'], tgt_data['origin_label']
            # origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            tgt_input = tgt_input.cuda(non_blocking=True)

            for i in range(len(origin_mask)):

                active_mask = origin_mask[i].cuda(non_blocking=True)
                ground_truth = origin_label[i].cuda(non_blocking=True)
                # size = (origin_size[i][0], origin_size[i][1])
                active = active_indicator[i]
                selected = selected_indicator[i]

                
                
                input_image = tgt_input[0].permute(1,2,0).cpu().numpy()
                # superpixels_snic_filename = snic(name[0], numsuperpixels=1024, compactness=0.1, image=input_image)
                # superpixels_labels, numlabels = segment(name[0], numsuperpixels=cfg.ACTIVE.SPIXELS, compactness=cfg.ACTIVE.COMPACTNESS, doRGBtoLAB=True, image=input_image)
                superpixels_labels = slic(load_image(name[0]), n_segments=cfg.ACTIVE.SPIXELS, compactness=cfg.ACTIVE.COMPACTNESS)
                numlabels = len(np.unique(superpixels_labels))
                # get superpixels gt labels
                # print(numlabels)
                gt = np.array(ground_truth.cpu())
                for sp_idx in range(numlabels):
                    # superpixels = superpixels_labels == i
                    sp_i = np.where(superpixels_labels == sp_idx)
                    selected_labels = gt[sp_i]
                    labels_in_sp, counts = np.unique(selected_labels, return_counts=True)
                    label_counts = dict(zip(labels_in_sp, counts))
                    if 255 in label_counts.keys():
                        del label_counts[255]
                        
                    active[sp_i] = True
                    selected[sp_i] = True
                    active_mask[sp_i] = labels_in_sp[np.argsort(counts)[-1]]
                
                active_mask = Image.fromarray(np.array(active_mask.cpu().numpy(), dtype=np.uint8))
                active_mask.save(path2mask[i])
                indicator = {
                    'active': active,
                    'selected': selected
                }
                torch.save(indicator, path2indicator[i])

    # feature_extractor.train()
    # classifier.train()
    
    
def enquiry_superpixels_label(superpixels_labels, sp_idx, gt):
    sp_i = np.where(superpixels_labels == sp_idx)
    selected_labels = gt[sp_i]
    labels_in_sp, counts = np.unique(selected_labels, return_counts=True)
    label_counts = dict(zip(labels_in_sp, counts))
    if 255 in label_counts.keys():
        del label_counts[255]
    return sp_i, labels_in_sp[np.argsort(counts)[-1]]


def generate_RIPU_scores(tgt_out, i, size, cfg, calculate_purity, active):
    output = tgt_out[i:i + 1, :, :, :]
    output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
    output = output.squeeze(dim=0)
    p = torch.softmax(output, dim=0)
    entropy = torch.sum(-p * torch.log(p + 1e-6), dim=0)
    pseudo_label = torch.argmax(p, dim=0)
    one_hot = F.one_hot(pseudo_label, num_classes=cfg.MODEL.NUM_CLASSES).float()
    one_hot = one_hot.permute((2, 0, 1)).unsqueeze(dim=0)
    purity = calculate_purity(one_hot).squeeze(dim=0).squeeze(dim=0)
    score = entropy * purity
    score[active] = -float('inf')
    return score


def generate_PU_scores(tgt_out, i, size, active):
    output = tgt_out[i:i + 1, :, :, :]
    output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
    output = output.squeeze(dim=0)
    p = torch.softmax(output, dim=0)
    entropy = torch.sum(-p * torch.log(p + 1e-6), dim=0)
    # pseudo_label = torch.argmax(p, dim=0)
    # one_hot = F.one_hot(pseudo_label, num_classes=cfg.MODEL.NUM_CLASSES).float()
    # one_hot = one_hot.permute((2, 0, 1)).unsqueeze(dim=0)
    # purity = calculate_purity(one_hot).squeeze(dim=0).squeeze(dim=0)
    score = entropy 
    score[active] = -float('inf')
    return score


def generate_RI_scores(tgt_out, i, size, cfg, calculate_purity, active):
    output = tgt_out[i:i + 1, :, :, :]
    output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
    output = output.squeeze(dim=0)
    p = torch.softmax(output, dim=0)
    # entropy = torch.sum(-p * torch.log(p + 1e-6), dim=0)
    pseudo_label = torch.argmax(p, dim=0)
    one_hot = F.one_hot(pseudo_label, num_classes=cfg.MODEL.NUM_CLASSES).float()
    one_hot = one_hot.permute((2, 0, 1)).unsqueeze(dim=0)
    purity = calculate_purity(one_hot).squeeze(dim=0).squeeze(dim=0)
    score = purity
    score[active] = -float('inf')
    return score
    
def generate_RIPU_superpixels_scores(ground_truth, superpixels_labels, numlabels, score):
    gt = np.array(ground_truth.cpu())
    sp_index, spa_labels = [], []
    scores = torch.zeros(numlabels)
    for sp_idx in range(numlabels):
        
        sp_i, spa_label = enquiry_superpixels_label(superpixels_labels, sp_idx, gt)
        score_i = score[sp_i].mean().item()
        sp_index.append(sp_i)
        spa_labels.append(spa_label)
        # scores.append(score_i)
        scores[sp_idx] = score_i  
    return sp_index, spa_labels, scores

  
def generate_SPA_superpixels_scores(ground_truth, superpixels_labels, numlabels, score):
    gt = np.array(ground_truth.cpu())
    sp_index, spa_labels = [], []
    scores = torch.zeros(numlabels)
    for sp_idx in range(numlabels):
        
        sp_i, spa_label = enquiry_superpixels_label(superpixels_labels, sp_idx, gt)
        score_i = score[sp_i].mean().item()
        sp_index.append(sp_i)
        spa_labels.append(spa_label)
        # scores.append(score_i)
        scores[sp_idx] = score_i  
    return sp_index, spa_labels, scores


def generate_active_mask(active_pixels, scores, sp_index, spa_labels, active, selected, active_mask, method="SPARIPU", allpixnum=1000):
    if method == "SPARANDOM":
        import random
        rnd_idx = random.sample(range(allpixnum), active_pixels)
    for pixel in range(active_pixels):
        if method == "SPARIPU" or method == "SPAPU" or method == "SPARI" or method == "SPARIW":
            values, indices_h = torch.max(scores, dim=0)
        elif method == "SPALL":
            indices_h = pixel
        elif method == "SPARANDOM":
            indices_h = rnd_idx[pixel]
        else:
            indices_h = 0
        # _, indices_w = torch.max(values, dim=0)
        # w = indices_w.item()
        # h = indices_h[w].item()

        # start_w = w - mask_radius if w - mask_radius >= 0 else 0
        # start_h = h - mask_radius if h - mask_radius >= 0 else 0
        # end_w = w + mask_radius + 1
        # end_h = h + mask_radius + 1
        
        sp_i = sp_index[indices_h]
        spa_label = spa_labels[indices_h]
        # active[sp_i] = True
        
        # active_mask[sp_i] = spa_label
        
        
        # mask out
        # score[start_h:end_h, start_w:end_w] = -float('inf')
        # active[start_h:end_h, start_w:end_w] = True
        # selected[h, w] = True
        if method == "SPARIPU" or method == "SPAPU" or method == "SPARI":
            scores[indices_h] = -float('inf')
        
        active[sp_i] = True
        selected[sp_i] = True
        
        # active sampling
        # active_mask[h, w] = ground_truth[h, w]
        active_mask[sp_i] = spa_label
    return active, selected, active_mask, scores
         
def SuperPixelRIPUSelection(cfg, feature_extractor, classifier, tgt_epoch_loader):
    feature_extractor.eval()
    classifier.eval()

    # SELECTED_PIXELS / SELECTED_ITER_TIMES / TARGET_INPUT_SIZE * TARGET_ORIGINAL_SIZE
    # for example:
    # from source_dataset GTAV to Cityscapes
    # 40 pixels / len([10000, 12000, 14000, 16000, 18000]) / (1280 * 640) * (2048 * 1024)
    # GTAV original image size: (1914 * 1052)
    # active_pixels = math.ceil(cfg.ACTIVE.PIXELS / len(cfg.ACTIVE.SELECT_ITER) / (1280 * 640) * (2048 * 1024))
    calculate_purity = SpatialPurity(in_channels=cfg.MODEL.NUM_CLASSES, size=2 * cfg.ACTIVE.RADIUS_K + 1).cuda()
    # mask_radius = cfg.ACTIVE.RADIUS_K

    with torch.no_grad():
        for tgt_data in tqdm(tgt_epoch_loader):

            tgt_input, path2mask, name = tgt_data['img'], tgt_data['path_to_mask'], tgt_data['name']
            origin_mask, origin_label = tgt_data['origin_mask'], tgt_data['origin_label']
            origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            tgt_input = tgt_input.cuda(non_blocking=True)

            tgt_size = tgt_input.shape[-2:]
            tgt_feat = feature_extractor(tgt_input)
            tgt_out = classifier(tgt_feat, size=tgt_size)

            for i in range(len(origin_mask)):

                active_mask = origin_mask[i].cuda(non_blocking=True)
                ground_truth = origin_label[i].cuda(non_blocking=True)
                size = (origin_size[i][0], origin_size[i][1])
                active = active_indicator[i]
                selected = selected_indicator[i]

                #*************************generate ripu score*****************************
                # score: (512, 512)
                # output = tgt_out[i:i + 1, :, :, :]
                # output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
                # output = output.squeeze(dim=0)
                # p = torch.softmax(output, dim=0)
                # entropy = torch.sum(-p * torch.log(p + 1e-6), dim=0)
                # pseudo_label = torch.argmax(p, dim=0)
                # one_hot = F.one_hot(pseudo_label, num_classes=cfg.MODEL.NUM_CLASSES).float()
                # one_hot = one_hot.permute((2, 0, 1)).unsqueeze(dim=0)
                # purity = calculate_purity(one_hot).squeeze(dim=0).squeeze(dim=0)
                # score = entropy * purity

                # score[active] = -float('inf')

                score = generate_RIPU_scores(tgt_out, i, size, cfg, calculate_purity, active)
                #***************************generate superpixels********************************
                # input_image = tgt_input[0].permute(1,2,0).cpu().numpy()
                # superpixels_snic_filename = snic(name[0], numsuperpixels=1024, compactness=0.1, image=input_image)
                imgpath = os.path.join("datasets/VOCdevkit/VOC2012/JPEGImages", name[0] + ".jpg")
                # superpixels_labels, numlabels = segment(imgpath, numsuperpixels=cfg.ACTIVE.SPALLPIXELS, compactness=cfg.ACTIVE.COMPACTNESS, doRGBtoLAB=True)
                superpixels_labels = slic(load_image(imgpath), n_segments=cfg.ACTIVE.SPIXELS, compactness=cfg.ACTIVE.COMPACTNESS)
                numlabels = len(np.unique(superpixels_labels))
                # get superpixels gt labels
                # print(numlabels)
                
                #****************************generate RIPU superpixels mean**************************************************
                sp_index, spa_labels, scores = generate_RIPU_superpixels_scores(ground_truth, superpixels_labels, numlabels, score)
                # sp_index, spa_labels = [], []
                # scores = torch.zeros(numlabels)
                # for sp_idx in range(numlabels):
                #     # superpixels = superpixels_labels == i
                #     # sp_i = np.where(superpixels_labels == sp_idx)
                #     # selected_labels = gt[sp_i]
                #     # labels_in_sp, counts = np.unique(selected_labels, return_counts=True)
                #     # label_counts = dict(zip(labels_in_sp, counts))
                #     # if 255 in label_counts.keys():
                #     #     del label_counts[255]
                #     sp_i, spa_label = enquiry_superpixels_label(superpixels_labels, sp_idx, gt)
                #     score_i = score[sp_i].mean().item()
                #     sp_index.append(sp_i)
                #     spa_labels.append(spa_label)
                #     # scores.append(score_i)
                #     scores[sp_idx] = score_i 
                    
                #********************************select superpixels by score**************************
                active_pixels = math.ceil(cfg.ACTIVE.PIXELS / len(cfg.ACTIVE.SELECT_ITER) / (cfg.INPUT.TARGET_INPUT_SIZE_TRAIN[0] * cfg.INPUT.TARGET_INPUT_SIZE_TRAIN[1]) * (tgt_input.shape[-2:][0] * tgt_input.shape[-2:][0]))
                active, selected, active_mask = generate_active_mask(active_pixels, scores, sp_index, spa_labels, active, selected, active_mask)
                

                active_mask = Image.fromarray(np.array(active_mask.cpu().numpy(), dtype=np.uint8))
                active_mask.save(path2mask[i])
                indicator = {
                    'active': active,
                    'selected': selected
                }
                torch.save(indicator, path2indicator[i])

    feature_extractor.train()
    classifier.train()


def generate_superpixels_idx(ground_truth, superpixels_labels, numlabels):
    gt = np.array(ground_truth.cpu())
    sp_index, spa_labels = [], []
    # scores = torch.zeros(numlabels)
    for sp_idx in range(numlabels):
        sp_i, spa_label = enquiry_superpixels_label(superpixels_labels, sp_idx, gt)
        # score_i = score[sp_i].mean().item()
        sp_index.append(sp_i)
        spa_labels.append(spa_label)
        # scores.append(score_i)
        # scores[sp_idx] = score_i  
    return sp_index, spa_labels

                   
def SuperPixelActiveSelection(cfg, feature_extractor, classifier, tgt_epoch_loader):
    feature_extractor.eval()
    classifier.eval()

    # SELECTED_PIXELS / SELECTED_ITER_TIMES / TARGET_INPUT_SIZE * TARGET_ORIGINAL_SIZE
    # for example:
    # from source_dataset GTAV to Cityscapes
    # 40 pixels / len([10000, 12000, 14000, 16000, 18000]) / (1280 * 640) * (2048 * 1024)
    # GTAV original image size: (1914 * 1052)
    # active_pixels = math.ceil(cfg.ACTIVE.PIXELS / len(cfg.ACTIVE.SELECT_ITER) / (1280 * 640) * (2048 * 1024))
    calculate_purity = SpatialPurity(in_channels=cfg.MODEL.NUM_CLASSES, size=2 * cfg.ACTIVE.RADIUS_K + 1).cuda()
    # mask_radius = cfg.ACTIVE.RADIUS_K

    with torch.no_grad():
        for tgt_data in tqdm(tgt_epoch_loader):

            tgt_input, path2mask, name = tgt_data['img'], tgt_data['path_to_mask'], tgt_data['name']
            origin_mask, origin_label = tgt_data['origin_mask'], tgt_data['origin_label']
            origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            tgt_input = tgt_input.cuda(non_blocking=True)

            tgt_size = tgt_input.shape[-2:]
            tgt_feat = feature_extractor(tgt_input)
            tgt_out = classifier(tgt_feat, size=tgt_size)

            for i in range(len(origin_mask)):

                active_mask = origin_mask[i].cuda(non_blocking=True)
                ground_truth = origin_label[i].cuda(non_blocking=True)
                size = (origin_size[i][0], origin_size[i][1])
                active = active_indicator[i]
                selected = selected_indicator[i]

                if 'cityscapes' in cfg.DATASETS.TARGET_TRAIN:
                    h, w = ground_truth.shape[0], ground_truth.shape[1]
                    imgpath = os.path.join("datasets/cityscapes/leftImg8bit/train", name[0])
                else:
                    w, h = cfg.INPUT.TARGET_INPUT_SIZE_TRAIN[0], cfg.INPUT.TARGET_INPUT_SIZE_TRAIN[1]
                    imgpath = os.path.join("datasets/VOCdevkit/VOC2012/JPEGImages", name[0] + ".jpg")
                    
                if cfg.ACTIVE.SETTING == "PARIPU" or cfg.ACTIVE.SETTING == "SPARIPU":
                    #*************************generate ripu score*****************************
                    score = generate_RIPU_scores(tgt_out, i, size, cfg, calculate_purity, active)
                    
                    #***************************generate superpixels********************************
                    # imgpath = os.path.join("datasets/VOCdevkit/VOC2012/JPEGImages", name[0] + ".jpg")
                    # superpixels_labels, numlabels = segment(imgpath, numsuperpixels=cfg.ACTIVE.SPALLPIXELS, compactness=cfg.ACTIVE.COMPACTNESS, doRGBtoLAB=True, resize=True, h=h, w=w)
                    superpixels_labels = slic(load_image(imgpath), n_segments=cfg.ACTIVE.SPIXELS, compactness=cfg.ACTIVE.COMPACTNESS)
                    numlabels = len(np.unique(superpixels_labels))
                    #****************************generate RIPU superpixels mean**************************************************
                    sp_index, spa_labels, scores = generate_RIPU_superpixels_scores(ground_truth, superpixels_labels, numlabels, score)
                                    
                    #********************************select superpixels by score**************************
                    active_pixels = math.ceil(cfg.ACTIVE.PIXELS / len(cfg.ACTIVE.SELECT_ITER) / (cfg.INPUT.TARGET_INPUT_SIZE_TRAIN[0] * cfg.INPUT.TARGET_INPUT_SIZE_TRAIN[1]) * (tgt_input.shape[-2:][0] * tgt_input.shape[-2:][0]))
                    active, selected, active_mask, scores = generate_active_mask(active_pixels, scores, sp_index, spa_labels, active, selected, active_mask)
                    
                elif cfg.ACTIVE.SETTING == "SPALL":
                    #***************************generate superpixels********************************
                    # imgpath = os.path.join("datasets/VOCdevkit/VOC2012/JPEGImages", name[0] + ".jpg")
                    # superpixels_labels, numlabels = segment(imgpath, numsuperpixels=cfg.ACTIVE.SPALLPIXELS, compactness=cfg.ACTIVE.COMPACTNESS, doRGBtoLAB=True, resize=True, h=h, w=w)
                    superpixels_labels = slic(load_image(imgpath), n_segments=cfg.ACTIVE.SPIXELS, compactness=cfg.ACTIVE.COMPACTNESS)
                    numlabels = len(np.unique(superpixels_labels))
                    #********************************select superpixels by score**************************
                    active_pixels = min(cfg.ACTIVE.SPALLPIXELS, numlabels)
                    sp_index, spa_labels = generate_superpixels_idx(ground_truth, superpixels_labels, numlabels)
                    active, selected, active_mask, scores = generate_active_mask(active_pixels, None, sp_index, spa_labels, active, selected, active_mask, method=cfg.ACTIVE.SETTING)
                elif cfg.ACTIVE.SETTING == "SPARANDOM":
                    #***************************generate superpixels********************************
                    # imgpath = os.path.join("datasets/VOCdevkit/VOC2012/JPEGImages", name[0] + ".jpg")
                    # superpixels_labels, numlabels = segment(imgpath, numsuperpixels=cfg.ACTIVE.SPALLPIXELS, compactness=cfg.ACTIVE.COMPACTNESS, doRGBtoLAB=True, resize=True, h=h, w=w)
                    superpixels_labels = slic(load_image(imgpath), n_segments=cfg.ACTIVE.SPIXELS, compactness=cfg.ACTIVE.COMPACTNESS)
                    numlabels = len(np.unique(superpixels_labels))
                    #********************************select superpixels by score**************************
                    active_pixels = math.ceil(cfg.ACTIVE.PIXELS / len(cfg.ACTIVE.SELECT_ITER) / (cfg.INPUT.TARGET_INPUT_SIZE_TRAIN[0] * cfg.INPUT.TARGET_INPUT_SIZE_TRAIN[1]) * (tgt_input.shape[-2:][0] * tgt_input.shape[-2:][0]))
                    sp_index, spa_labels = generate_superpixels_idx(ground_truth, superpixels_labels, numlabels)                    
                    active, selected, active_mask, scores = generate_active_mask(active_pixels, None, sp_index, spa_labels, active, selected, active_mask, method=cfg.ACTIVE.SETTING, allpixnum=min(cfg.ACTIVE.SPALLPIXELS, numlabels))
                elif cfg.ACTIVE.SETTING == "SPAPU":
                    #*************************generate ripu score*****************************
                    score = generate_PU_scores(tgt_out, i, size, active)
                    
                    #***************************generate superpixels********************************
                    # imgpath = os.path.join("datasets/VOCdevkit/VOC2012/JPEGImages", name[0] + ".jpg")
                    # superpixels_labels, numlabels = segment(imgpath, numsuperpixels=cfg.ACTIVE.SPALLPIXELS, compactness=cfg.ACTIVE.COMPACTNESS, doRGBtoLAB=True, resize=True, h=h, w=w)
                    superpixels_labels = slic(load_image(imgpath), n_segments=cfg.ACTIVE.SPIXELS, compactness=cfg.ACTIVE.COMPACTNESS)
                    numlabels = len(np.unique(superpixels_labels))
                    #****************************generate SPA superpixels mean**************************************************
                    sp_index, spa_labels, scores = generate_SPA_superpixels_scores(ground_truth, superpixels_labels, numlabels, score)
                                    
                    #********************************select superpixels by score**************************
                    active_pixels = math.ceil(cfg.ACTIVE.PIXELS / len(cfg.ACTIVE.SELECT_ITER) / (cfg.INPUT.TARGET_INPUT_SIZE_TRAIN[0] * cfg.INPUT.TARGET_INPUT_SIZE_TRAIN[1]) * (tgt_input.shape[-2:][0] * tgt_input.shape[-2:][0]))
                    active, selected, active_mask, scores = generate_active_mask(active_pixels, scores, sp_index, spa_labels, active, selected, active_mask)
                elif cfg.ACTIVE.SETTING == "SPARI" or cfg.ACTIVE.SETTING == "SPARIW":
                    #*************************generate ripu score*****************************
                    score = generate_RI_scores(tgt_out, i, size, cfg, calculate_purity, active)
                    
                    #***************************generate superpixels********************************
                    # imgpath = os.path.join("datasets/VOCdevkit/VOC2012/JPEGImages", name[0] + ".jpg")
                    # superpixels_labels, numlabels = segment(imgpath, numsuperpixels=cfg.ACTIVE.SPALLPIXELS, compactness=cfg.ACTIVE.COMPACTNESS, doRGBtoLAB=True, resize=True, h=h, w=w)
                    superpixels_labels = slic(load_image(imgpath), n_segments=cfg.ACTIVE.SPIXELS, compactness=cfg.ACTIVE.COMPACTNESS)
                    numlabels = len(np.unique(superpixels_labels))
                    #****************************generate SPA superpixels mean**************************************************
                    sp_index, spa_labels, scores = generate_SPA_superpixels_scores(ground_truth, superpixels_labels, numlabels, score)
                                    
                    #********************************select superpixels by score**************************
                    active_pixels = math.ceil(cfg.ACTIVE.PIXELS / len(cfg.ACTIVE.SELECT_ITER) / (cfg.INPUT.TARGET_INPUT_SIZE_TRAIN[0] * cfg.INPUT.TARGET_INPUT_SIZE_TRAIN[1]) * (tgt_input.shape[-2:][0] * tgt_input.shape[-2:][0]))
                    active, selected, active_mask, scores = generate_active_mask(active_pixels, scores, sp_index, spa_labels, active, selected, active_mask)
                    
                            
                active_mask = Image.fromarray(np.array(active_mask.cpu().numpy(), dtype=np.uint8))
                active_mask.save(path2mask[i])
                indicator = {
                    'active': active,
                    'selected': selected
                }
                torch.save(indicator, path2indicator[i])

    feature_extractor.train()
    classifier.train()


def EntitySelection(cfg, tgt_epoch_loader):
 
    with torch.no_grad():
        # tgt_epoch_loader with fixed batch_size=1
        for tgt_data in tqdm(tgt_epoch_loader):

            tgt_input, path2mask, name = tgt_data['img'], tgt_data['path_to_mask'], tgt_data['name']
            origin_mask, origin_label = tgt_data['origin_mask'], tgt_data['origin_label']
            # origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            tgt_input = tgt_input.cuda(non_blocking=True)

            for i in range(len(origin_mask)):

                active_mask = origin_mask[i].cuda(non_blocking=True)
                ground_truth = origin_label[i].cuda(non_blocking=True)
                # size = (origin_size[i][0], origin_size[i][1])
                active = active_indicator[i]
                selected = selected_indicator[i]

                superpixels_labels = slic(load_image(name[0]), n_segments=cfg.ACTIVE.SPIXELS, compactness=cfg.ACTIVE.COMPACTNESS)
                numlabels = len(np.unique(superpixels_labels))
                # get superpixels gt labels
                # print(numlabels)
                gt = np.array(ground_truth.cpu())
                for sp_idx in range(numlabels):
                    # superpixels = superpixels_labels == i
                    sp_i = np.where(superpixels_labels == sp_idx)
                    selected_labels = gt[sp_i]
                    labels_in_sp, counts = np.unique(selected_labels, return_counts=True)
                    label_counts = dict(zip(labels_in_sp, counts))
                    if 255 in label_counts.keys():
                        del label_counts[255]
                        
                    active[sp_i] = True
                    selected[sp_i] = True
                    active_mask[sp_i] = labels_in_sp[np.argsort(counts)[-1]]
                
                active_mask = Image.fromarray(np.array(active_mask.cpu().numpy(), dtype=np.uint8))
                active_mask.save(path2mask[i])
                indicator = {
                    'active': active,
                    'selected': selected
                }
                torch.save(indicator, path2indicator[i])



def EntitySegSelection(cfg, tgt_epoch_loader):
    with torch.no_grad():
        for tgt_data in tqdm(tgt_epoch_loader):

            tgt_input, path2mask = tgt_data['img'], tgt_data['path_to_mask']
            origin_mask, origin_label = \
                tgt_data['origin_mask'], tgt_data['origin_label']
            origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']
            
            label_es = tgt_data['label_es']
            
          
            
            for i in range(len(origin_mask)):
                active = ~active_indicator[i]
                selected = ~selected_indicator[i]
                active_mask = label_es[i]
          
            active_mask = Image.fromarray(np.array(active_mask.cpu().numpy(), dtype=np.uint8))
            active_mask.save(path2mask[i])
            indicator = {
                'active': active,
                'selected': selected
            }
            torch.save(indicator, path2indicator[i])


def CORESETSelection(cfg, feature_extractor, classifier, tgt_epoch_loader):
    feature_extractor.eval()
    classifier.eval()


    with torch.no_grad():
        # for tgt_data in tqdm(tgt_epoch_loader):
        # **********!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for index, tgt_data in tqdm(enumerate(tgt_epoch_loader)):
            # if the image is labeled, skip it
            # target epoch loader with fixed batch_size=1,
            # print(index)
            if index < cfg.ACTIVE.LABELED:
                continue

            # tgt_input, path2mask = tgt_data['img'], tgt_data['path_to_mask']
            path2mask = tgt_data['path_to_mask']
            print(path2mask)
            origin_mask, origin_label = tgt_data['origin_mask'], tgt_data['origin_label']
            # origin_mask = tgt_data['origin_mask']
            # tgt_data['origin_label']
            # origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            

            for i in range(len(origin_mask)):

                active_mask = origin_mask[i].cuda(non_blocking=True)
                # ground_truth = origin_label[i].cuda(non_blocking=True)
                # size = (origin_size[i][0], origin_size[i][1])
                active = active_indicator[i]
                selected = selected_indicator[i]


                ########################################################
                if "voc" in cfg.DATASETS.TARGET_TRAIN:
                    dataset = "voc"
                    npy_path = os.path.join("datasets/VOCdevkit/VOC2012/SegmentationClass_sam_npz", tgt_data['name'][0] + ".npz")
                elif "cityscapes" in cfg.DATASETS.TARGET_TRAIN:
                    dataset = "cityscapes"
                    npy_path = os.path.join("datasets/cityscapes/segmentation_sam_npz/train", tgt_data['name'][0].split("/")[-1].split(".")[0] + ".npz")
                start_time = time.time()
                active_mask = generate_sam_labels(npy_path, origin_label, cfg.ACTIVE.MASKS, dataset)
                end_time = time.time()
                print("generate sam labels time: ", end_time - start_time)
                active[active_mask!=255] = True
                selected[active_mask!=255] = True
                
                # active_mask = Image.fromarray(np.array(active_mask.cpu().numpy(), dtype=np.uint8))
                active_mask = Image.fromarray(active_mask)

                active_mask.save(path2mask[i])
                indicator = {
                    'active': active,
                    'selected': selected
                }
                torch.save(indicator, path2indicator[i])

    feature_extractor.train()
    classifier.train()
