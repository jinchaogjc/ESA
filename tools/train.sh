#!/bin/bash 

# training for COCO to VOC
export CUDA_VISIBLE_DEVICES=0
export output=results/deeplabv3plus_r101_cocovoc_PA_40_d0
python train_coco.py -cfg configs/coco2voc/deeplabv3plus_r101_cocovoc_PA_40_d0.yaml OUTPUT_DIR $output