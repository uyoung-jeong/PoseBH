#!/bin/bash
python -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_addr 127.0.0.1 --master_port 23459 tools/train.py configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mdt/vitb_posebh.py \
--cfg-options model.multihead_pretrained=weights/vitpose+_base.pth --launcher pytorch --seed 0
