#!/bin/bash
# usage example: ./scripts/infer_ap10k_vitb.sh work_dirs/vitPose+_base_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp mh 10
weight_dir=$1
name=$2
freq=${3-1}
skeleton_only=${4-0}
return_heatmap=${5-0}
vis_pdf=${6-1}

# infer with ap10k keypoints
python tools/inference.py --vis_freq $freq \
    --model_config configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_base_ap10k_256x192.py \
    --dataset_config configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_base_ap10k_256x192.py \
    --checkpoint $weight_dir/ap10k.pth --model_name $name'_ap10k' --device 'cuda:0' \
    --skeleton_only $skeleton_only --return_heatmap $return_heatmap --vis_pdf $vis_pdf

# infer with apt36k keypoints
python tools/inference.py --vis_freq $freq \
    --model_config configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/apt36k/ViTPose_base_apt36k_256x192.py \
    --dataset_config configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_base_ap10k_256x192.py \
    --checkpoint $weight_dir/apt36k.pth --model_name $name'_apt36k' --device 'cuda:1' \
    --skeleton_only $skeleton_only --return_heatmap $return_heatmap --vis_pdf $vis_pdf
