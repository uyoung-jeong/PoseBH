#!/bin/bash
# usage example: 
# ./scripts/infer_coco_vitb.sh work_dirs/vitPose+_base_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp mh 10 1 1
weight_dir=$1
name=$2
freq=${3-100}
skeleton_only=${4-0}
return_heatmap=${5-0}
vis_pdf=${6-1}

# infer with coco keypoints
python tools/inference.py --vis_freq $freq \
    --model_config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192_gteval.py \
    --dataset_config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192_gteval.py \
    --checkpoint $weight_dir/coco.pth --model_name $name'_coco' --device 'cuda:0' \
    --skeleton_only $skeleton_only --return_heatmap $return_heatmap --vis_pdf $vis_pdf

# infer with aic keypoints
python tools/inference.py --vis_freq $freq \
    --model_config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/aic/ViTPose_base_aic_256x192.py \
    --dataset_config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192_gteval.py \
    --checkpoint $weight_dir/aic.pth --model_name $name'_aic' --device 'cuda:1' \
    --skeleton_only $skeleton_only --return_heatmap $return_heatmap --vis_pdf $vis_pdf

# infer with mpii keypoints
python tools/inference.py --vis_freq $freq \
    --model_config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/ViTPose_base_mpii_256x192.py \
    --dataset_config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192_gteval.py \
    --checkpoint $weight_dir/mpii.pth --model_name $name'_mpii' --device 'cuda:0' \
    --skeleton_only $skeleton_only --return_heatmap $return_heatmap --vis_pdf $vis_pdf

# infer with coco-wholebody keypoints
python tools/inference.py --vis_freq $freq \
    --model_config configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_base_wholebody_256x192.py \
    --dataset_config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192_gteval.py \
    --checkpoint $weight_dir/wholebody.pth --model_name $name'_wholebody' --device 'cuda:1' \
    --skeleton_only $skeleton_only --return_heatmap $return_heatmap --vis_pdf $vis_pdf
