# infer pose without evaluation
"""
Usage examples
eval on coco with coco model:
python tools/inference.py --vis_freq 100 \
    --model_config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192_gteval.py \
    --dataset_config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192_gteval.py \
    --checkpoint work_dirs/vitPose+_base_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp/coco.pth \
    --model_name mh_coco \
    --device 'cuda:0' \
    --skeleton_only 0 --return_heatmap 1

eval on coco with aic model:
python tools/inference.py --vis_freq 100 \
    --model_config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/aic/ViTPose_base_aic_256x192.py \
    --dataset_config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192_gteval.py \
    --checkpoint work_dirs/vitPose+_base_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp/aic.pth \
    --model_name mh_aic \
    --device 'cuda:1' --skeleton_only 0 --return_heatmap 1
"""
import os
import os.path as osp
import warnings
from argparse import ArgumentParser
from functools import partial

import requests

import torch

from mmcv.parallel import collate

from mmcv import Config, DictAction
from mmpose.datasets import build_dataloader, build_dataset
from mmcv.utils.parrots_wrapper import _get_dataloader

from mmpose.apis import (inference_bottom_up_pose_model,
                         inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.models import AssociativeEmbedding, TopDown, TopDownProto
from mmpose.core.post_processing import get_affine_transform

from glob import glob
from tqdm import tqdm
import numpy as np
import cv2

from tools.coco_infer_img_files import coco_img_files
from tools.ap10k_infer_img_files import ap10k_img_files
from tools.pw3d_infer_img_files import pw3d_img_files
from tools.interhand_infer_img_files import interhand_img_files
from tools.posetrack_img_files import posetrack_img_files
from tools.crowdpose_img_files import crowdpose_img_files

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input_img', help='Image file for inference', default='')
    parser.add_argument('--input_dir', help='Directory for inference', default='')
    parser.add_argument('--vis_freq', help='frequency for visualization. only used when input_dir is given', default=1, type=int)
    parser.add_argument('--model_config', help='Model config file', default='')
    parser.add_argument('--dataset_config', help='Dataset config file', default='')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--model_name', help='The model name in the server')
    parser.add_argument('--skeleton_only', help='Whether to visualize skeleton without img.', type=int, default=0)
    parser.add_argument('--return_heatmap', help='Whether to return keypoint heatmap', type=int, default=0)
    #parser.add_argument('--dataset_name', help='dataset name for skeleton visualization', default='')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the dataset config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--out_dir', default='vis_results', help='Visualization output path')
    parser.add_argument(
        '--vis_pdf', default=1, type=int, help='Visualize as pdf format or not')
    args = parser.parse_args()
    return args

# derived from mmpose.apis.inference.py
color_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255],
                            [255, 0, 0], [255, 255, 255], [84, 28, 0]])

color_dict = {
    'coco': color_palette[[9, 9, 9, 9, 9, 16, 0, 16, 0, 16, 0, 16, 0, 16, 0, 16, 0]],
    'aic': color_palette[[0, 0, 0, 16, 16, 16, 0, 0, 0, 16, 16, 16, 9, 9]],
    'mpii': color_palette[[0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 0, 0, 0, 16, 16, 16]],
    'ap10k': color_palette[[16, 0, 9, 9, 9, 9, 9, 16, 16, 0, 16, 0, 0, 16, 16, 16, 16]],
    'coco_wholebody': color_palette[[9, 9, 9, 9, 9, 16, 0, 16, 0, 16, 0, 16, 0, 16, 0, 16, 0, 0, 0, 0, 0, 0, 0, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16, 16, 20, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16, 16]],
    'interhand2d': color_palette[[9, 9, 9, 9, 16, 16, 16, 16, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 0]],
    'pw3d': color_palette[[17, 16, 0, 17, 16, 0, 17, 16, 0, 17, 16, 0, 9, 16, 0, 9, 16, 0, 16, 0, 16, 0, 16, 0]],
    'posetrack18': color_palette[[9, 9, 9, 9, 9, 16, 0, 16, 0, 16, 0, 16, 0, 16, 0, 16, 0]],
    'crowdpose': color_palette[[16, 0, 16, 0, 16, 0, 16, 0, 16, 0, 16, 0, 9, 9]],
}

dataset_idx_dict = {
    'coco': 0,
    'aic': 1,
    'mpii': 2,
    'ap10k': 3,
    'apt36k': 4, #  this is actually not used
    'coco_wholebody': 5,
    'interhand2d': 0, # transfer dataset
    'pw3d': 0, # transfer dataset
    'posetrack18': 0, # transfer dataset
    'corwdpose': 0 # transfer dataset
}

def get_dataset_name(dataset):
    dataset_name = str(dataset)
    if 'TopDownCocoDataset' in dataset_name:
        dataset_name = 'TopDownCocoDataset'
    elif 'TopDownAicDataset' in dataset_name:
        dataset_name = 'TopDownAicDataset'
    elif 'TopDownMpiiDataset' in dataset_name:
        dataset_name = 'TopDownMpiiDataset'
    elif 'AnimalAP10KDataset' in dataset_name:
        dataset_name = 'AnimalAP10KDataset'
    elif 'TopDownCocoWholeBodyDataset' in dataset_name:
        dataset_name = 'TopDownCocoWholeBodyDataset'
    elif 'InterHand2DDataset' in dataset_name:
        dataset_name = 'InterHand2DDataset'
    elif 'TopDownPW3DDataset' in dataset_name:
        dataset_name = 'TopDownPW3DDataset'
    elif 'TopDownPoseTrack18Dataset' in dataset_name:
        dataset_name = 'TopDownPoseTrack18Dataset'
    elif 'TopDownCrowdPoseDataset' in dataset_name:
        dataset_name = 'TopDownCrowdPoseDataset'
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset_name}")
    return dataset_name

def check_oob(kpts, h, w):
    x_valid = (kpts[:,0] > 0) * (kpts[:,0]< w)
    y_valid = (kpts[:,1] > 0) * (kpts[:,1]< h)
    return x_valid * y_valid

def run_single_inference(args, model, dataset, img, meta=None, person_results=None, return_heatmap=False, skeleton_only=0):
    dataset_name = get_dataset_name(dataset)
    
    model_dset_name = model.cfg.dataset_info.dataset_name
    kpt_colors = color_dict[model_dset_name]
    dataset_idx = dataset_idx_dict[model_dset_name]

    if isinstance(model, (TopDown, TopDownProto):
        pytorch_result, hm_res = inference_top_down_pose_model(
            model, img, person_results=person_results, return_heatmap=return_heatmap,
            dataset_idx=dataset_idx)
    elif isinstance(model, (AssociativeEmbedding, )):
        pytorch_result, hm_res = inference_bottom_up_pose_model(model, img)
    else:
        print(f"model type not recognized. type(model): {type(model)}")
        raise NotImplementedError()
    

    conf_thr = 0.2
    #radius = 3
    #thickness = 2
    radius = 4
    thickness = 3

    out_dir = args.out_dir
    if dataset_name == 'TopDownPW3DDataset':
        save_img_name = '_'.join(img.split('/')[-2:]).replace('.jpg', f'_{args.model_name}.jpg')
        pdf_img_name = '_'.join(img.split('/')[-2:]).replace('.jpg', f'_{args.model_name}.pdf')
        jpg_img_name = '_'.join(img.split('/')[-2:])
    elif dataset_name == 'TopDownPoseTrack18Dataset':
        save_img_name = '_'.join(img.split('/')[-2:]).replace('.jpg', f'_{args.model_name}.jpg')
        pdf_img_name = '_'.join(img.split('/')[-2:]).replace('.jpg', f'_{args.model_name}.pdf')
        jpg_img_name = '_'.join(img.split('/')[-2:])
    else:
        save_img_name = img.split('/')[-1].replace('.jpg', f'_{args.model_name}.jpg').replace('.png', f'_{args.model_name}.png')
        pdf_img_name = img.split('/')[-1].replace('.jpg', f'_{args.model_name}.pdf').replace('.png', f'_{args.model_name}.pdf')
        jpg_img_name = img.split('/')[-1]
    
    vis_pdf = args.vis_pdf
    out_img_name = pdf_img_name if vis_pdf>0 else jpg_img_name

    img_rgb = cv2.imread(img)    
    H, W = img_rgb.shape[:2]
    # crop img
    img_crop = None
    if meta is not None:
        c = meta['center']
        s = meta['scale']
        r = 0
        input_img_size = [192, 256]
        trans = get_affine_transform(c, s, r, input_img_size)
        img_crop = cv2.warpAffine(
                    img_rgb,
                    trans, (int(input_img_size[0]), int(input_img_size[1])),
                    flags=cv2.INTER_LINEAR)
    if skeleton_only > 0:
        for res in pytorch_result:
            res.pop('bbox')
        img = np.zeros_like(img_rgb) + 255
        img = vis_pose_result(
            model,
            img,
            pytorch_result,
            radius=radius,
            thickness=thickness,
            kpt_score_thr=conf_thr,
            dataset=dataset_name,
            out_file=osp.join(out_dir, out_img_name))
    else:
        img = vis_pose_result(
            model,
            img,
            pytorch_result,
            radius=radius,
            thickness=thickness,
            kpt_score_thr=conf_thr,
            dataset=dataset_name,
            out_file=osp.join(out_dir, out_img_name))
    
    if len(hm_res[0]) > 0: # save heatmap
        hm = hm_res[0]['heatmap'][0]
        K, h, w = hm.shape

        # zero out unconfident keypoints
        hm_idxs = np.argmax(hm, axis=0)
        hm_max = np.amax(hm, axis=0, keepdims=True)
        hm_max = np.repeat(hm_max, hm.shape[0], axis=0)
        hm[hm_max < conf_thr] = 0

        len_kpt_colors = len(kpt_colors)
        if len_kpt_colors < K:
            idxs = [i%len_kpt_colors for i in range(K)]
            kpt_colors = kpt_colors[idxs]
        
        hm_vis = kpt_colors[:, None, None, :] * hm[:, :, :, None]

        hm_vis = np.take_along_axis(hm_vis, hm_idxs[None, :, :, None], axis=0).squeeze(0)
        hm_vis = np.clip(hm_vis, 0, 255).astype(np.uint8)

        if img_crop is not None:
            crop_shape = img_crop.shape[:2]
            hm_vis = cv2.resize(hm_vis, (crop_shape[1], crop_shape[0]))
            hm_vis = hm_vis * 0.8 + img_crop * 0.2
        
        hm_path = os.path.join(out_dir, save_img_name.replace('.jpg', '_hm.jpg').replace('.png', f'_hm.png'))
        cv2.imwrite(hm_path, hm_vis.astype(np.uint8)) # save heatmap
        

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    skeleton_only = args.skeleton_only
    return_heatmap = args.return_heatmap==1

    # Inference single image by native apis.
    model = init_pose_model(args.model_config, args.checkpoint, device=args.device)
    model_cfg = Config.fromfile(args.model_config)
    model_dataset = model_cfg.data.train.type
    model_dataset_name = get_dataset_name(model_dataset)

    ckpt = torch.load(args.checkpoint)
    weights = None
    if 'state_dict' in ckpt.keys():
        weights = ckpt['state_dict']
    else:
        weights = ckpt

    if args.dataset_config != '':
        dset_cfg = Config.fromfile(args.dataset_config)
        if args.cfg_options is not None:
            dset_cfg.merge_from_dict(args.cfg_options)
        dataset = build_dataset(dset_cfg.data.test, dict(test_mode=True))
        dataset_train = build_dataset(dset_cfg.data.test, dict(test_mode=False))
        
        _, DataLoader = _get_dataloader()
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=None,
            num_workers=4,
            collate_fn=partial(collate, samples_per_gpu=1),
            pin_memory=True,
            shuffle=False,
            worker_init_fn=None,
            drop_last=False,)
    else:
        dataset = None
        dataset_train = None
    
    if args.input_img != '':
        img = args.input_img
        res = run_single_inference(args, model, dataset, img, person_results=None)
    elif args.input_dir != '':
        dir_name = args.input_dir
        imgs = glob(f'{dir_name}/*.jpg') + glob(f'{dir_name}/*.png')
        imgs = sorted(imgs)
        for idx, img in enumerate(tqdm(imgs)):
            if idx % args.vis_freq == 0:
                res = run_single_inference(args, model, dataset, img, person_results=None)
    elif dataset is not None:
        n_dataset = len(dataset)
        images = {e['image_file']:i for i,e in enumerate(dataset.db)}
        dataset_name = get_dataset_name(dataset)
        print(f'parsed dataset name: {dataset_name}')

        palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255],
                            [255, 0, 0], [255, 255, 255]])
        if 'CrowdPose' in dataset_name:
            dset_cfg.data.test.dataset_info.skeleton = [[10, 8], [8, 6], [11, 9], [9, 7], [6, 7], 
                                                    [0, 6], [1, 7], [0, 1], [0, 2], [1, 3], 
                                                    [2, 4], [3, 5], [12, 13], [1, 13], [0, 13]]
            dset_cfg.data.test.dataset_info.pose_link_color = palette[[16, 16, 0, 0, 9, 9, 9, 9, 16, 0, 16, 0, 9, 9, 9]]
            dset_cfg.data.test.dataset_info.pose_kpt_color = palette[[16, 0, 16, 0, 16, 0, 16, 0, 16, 0, 16, 0, 9, 9]]

        for idx, batched_inputs in enumerate(tqdm(data_loader)):
            if idx % args.vis_freq != 0:
                continue
            
            inputs = batched_inputs['img_metas'].data[0]
            # get bbox. nasty but still works
            n_obj = len(inputs)
            if n_obj == 0:
                continue
            elif n_obj > 1:
                print(f"multiple objects detected. file name: {inputs[0]['image_file']}, num obj: {n_obj}")
                import ipdb; ipdb.set_trace()

            img = inputs[0]['image_file']

            if 'PW3D' in dataset_name:
                #img_trim = '/'.join(img.split('/')[-2:])
                img_trim = '_'.join(img.split('/')[-2:])
                if img_trim not in pw3d_img_files:
                    continue
            elif 'InterHand' in dataset_name:
                img_trim = '_'.join(img.split('/')[5:])
                
                if img_trim not in interhand_img_files:
                    continue
            elif 'PoseTrack' in dataset_name:
                img_trim = '/'.join(img.split('/')[-3:])
                #if img_trim not in posetrack_img_files:
                #    continue
            elif 'CrowdPose' in dataset_name:
                img_trim = img.split('/')[-1]
                if img_trim not in crowdpose_img_files:
                    continue
            elif 'AP10K' in dataset_name:
                img_name = img.split('/')[-1]
                if img_name not in ap10k_img_files:
                    continue
            elif 'coco' in dataset_name.lower():
                img_name = img.split('/')[-1]
                if img_name not in coco_img_files: # filter by img name
                    continue
            else:
                pass
            
            bboxs = []
            
            db_id = images[img]
            meta = dataset.db[db_id]
            bbox = dataset.db[db_id]['bbox']
            bboxs.append(bbox)

            if len(bboxs) == 0:
                continue

            # filter out small instances
            area = bbox[2] * bbox[3]
            if area < 3000:
                continue

            # filter out instances with few visible keypoints
            gt_vis = dataset.db[db_id]['joints_3d_visible']
            n_vis = gt_vis[:,0].sum()
            if n_vis < 6:
                continue
            
            img = inputs[0]['image_file']
            person_results = [{'bbox': bbox} for bbox in bboxs]
            res = run_single_inference(args, model, dataset, img, meta=meta,
                            person_results=person_results, 
                            return_heatmap=return_heatmap,
                            skeleton_only=skeleton_only)

            gt_kpts = []
            gt_kpt = dataset_train.db[db_id]['joints_3d']
            
            img_cv = cv2.imread(img)
            h, w = img_cv.shape[:2]
            gt_kpt[:,2] = check_oob(gt_kpt, h, w)
            gt_kpts.append(gt_kpt)
            if len(gt_kpts)>0 and model_dataset_name == dataset_name: # visualize gt
                gt_res = [{'keypoints': gt_kpt,
                        'bbox': bbox} for gt, bbox in zip(gt_kpts, bboxs)]
                save_img_name = img.split('/')[-1].replace('.jpg', '_gt.jpg').replace('.png', '_gt.png')
                pdf_img_name = img.split('/')[-1].replace('.jpg', '_gt.pdf').replace('.png', '_gt.pdf')

                gt_img = vis_pose_result(
                    model,
                    img,
                    gt_res,
                    radius=4,
                    thickness=3,
                    kpt_score_thr=0.2,
                    dataset=dataset_name,
                    #dataset_info=dset_cfg.data.test.dataset_info,
                    out_file=osp.join(args.out_dir, pdf_img_name))
            
    else:
        raise RuntimeError("input_img or input_dir should be specified.")

if __name__ == '__main__':
    main()
