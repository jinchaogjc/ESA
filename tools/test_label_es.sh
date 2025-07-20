#!/bin/bash 
python test.py \
    -cfg configs/gtav/deeplabv3plus_r101_RA_es.yaml \
    --mode "val" \
    resume results/v3plus_gtav_ra_5.0_precent_es/model_iter040000.pth \
    OUTPUT_DIR results/v3plus_gtav_ra_5.0_precent_es
