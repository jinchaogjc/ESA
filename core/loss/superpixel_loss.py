import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tools.utils import poolfeat, upfeat

'''
Loss function
author:Fengting Yang 
Mar.1st 2019

We only use "compute_semantic_pos_loss" func. in our final version, best result achieved with weight = 3e-3
'''

def compute_semantic_pos_loss(prob_in, labxy_feat,  pos_weight = 0.003,  kernel_size=16, target_mask=None):
    # this wrt the slic paper who used sqrt of (mse)

    # rgbxy1_feat: B*50+2*H*W
    # output : B*9*H*w
    # NOTE: this loss is only designed for one level structure

    # todo: currently we assume the downsize scale in x,y direction are always same
    S = kernel_size
    m = pos_weight
    # prob = prob_in.clone()

    b, c, h, w = labxy_feat.shape
    # pooled_labxy = poolfeat(labxy_feat, prob, kernel_size, kernel_size)
    # reconstr_feat = upfeat(pooled_labxy, prob, kernel_size, kernel_size)

    # pooled_labxy = pooled_labxy.clone()
    # reconstr_feat = reconstr_feat.clone()
    
    # loss_map = reconstr_feat[:,-2:,:,:] - labxy_feat[:,-2:,:,:]
    # TODO delete
    labxy_feat.clone()
    loss_map = torch.zeros((labxy_feat.size()[0],2,labxy_feat.size()[2],labxy_feat.size()[3])).cuda()
    # self def cross entropy  -- the official one combined softmax
    # logit = torch.log(reconstr_feat[:, :-2, :, :] + 1e-8)
    logit = torch.zeros((labxy_feat.size()[0],labxy_feat.size()[1],labxy_feat.size()[2],labxy_feat.size()[3])).cuda()
    
    if not target_mask is None:
        # target_mask = target_mask.to(torch.bool)
        target_mask = (target_mask == 0)
    target_mask1 = target_mask.unsqueeze(1).repeat_interleave(logit.size(1), dim=1)
    # maksed_feat = torch.zeros_like(logit)
    masked_feat = logit * labxy_feat[:, :-2, :, :]
    # masked_feat.masked_fill_(target_mask.unsqueeze(1), 0)
    masked_feat = masked_feat.clone()
    masked_feat.masked_fill_(target_mask1, 0)
    loss_sem = - torch.sum(masked_feat) / b
    target_mask2 = target_mask.unsqueeze(1).repeat_interleave(loss_map.size(1), dim=1)
    loss_map = loss_map.clone()
    loss_map.masked_fill_(target_mask2, 0)
    loss_pos = torch.norm(loss_map, p=2, dim=1).sum() / b * m / S

    # empirically we find timing 0.005 tend to better performance
    loss_sum =  0.005 * (loss_sem + loss_pos)
    loss_sem_sum =  0.005 * loss_sem
    loss_pos_sum = 0.005 * loss_pos

    return loss_sum, loss_sem_sum,  loss_pos_sum
