#!/bin/bash
weight_path=$1
if [ -z "$weight_path" ]; then
    echo "weight_path argument is not provided."
    exit 1
fi
weight_dir=$(dirname $weight_path)

python tools/model_split.py --source $weight_path

# evaluate coco
bash tools/dist_test.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py \
    $weight_dir/coco.pth 4
printf "\n------------------------\n"
# evaluate aic
bash tools/dist_test.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/aic/ViTPose_base_aic_256x192.py \
    $weight_dir/aic.pth 4
printf "\n------------------------\n"
# evaluate mpii
bash tools/dist_test.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/ViTPose_base_mpii_256x192.py \
    $weight_dir/mpii.pth 4
printf "\n------------------------\n"
# evaluate ap10k
bash tools/dist_test.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_base_ap10k_256x192.py \
    $weight_dir/ap10k.pth 4 --cfg-options data.test.ann_file=/home/uyoung/human/ViTPose/data/ap10k/annotations/ap10k-val-split1.json
printf "\n------------------------\n"
# evaluate ap36k
bash tools/dist_test.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/apt36k/ViTPose_base_apt36k_256x192.py \
    $weight_dir/apt36k.pth 4 --cfg-options data.test.ann_file=/home/uyoung/human/ViTPose/data/ap36k/annotations/apt36k_val.json
printf "\n------------------------\n"
# evaluate coco-wholebody
bash tools/dist_test.sh configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_base_wholebody_256x192.py \
    $weight_dir/wholebody.pth 4
