# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)

from mmpose.core.evaluation import pose_pck_accuracy
from mmpose.core.post_processing import flip_back
from mmpose.models.builder import build_loss
from mmpose.models.utils.ops import resize
from ..builder import HEADS
import torch.nn.functional as F
from .topdown_heatmap_base_head import TopdownHeatmapBaseHead

import torch.distributed as dist
from einops import rearrange
import math
from .kpt_proto_module import KptPrototype, ResBlock

from mmpose.core.evaluation.top_down_eval import _get_max_preds, post_dark_udp
from mmpose.datasets.pipelines.top_down_transform import TopDownGenerateTarget
import numpy as np
from mmpose.models.backbones.vit import DropPath

from mmcv.runner import get_dist_info

@HEADS.register_module()
class KptProtoHead(TopdownHeatmapBaseHead):
    """Top-down keypoint prototype head.

    KptProtoHead is consisted of (>=0) number of deconv layers
    and several layers by design choice.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        in_index (int|Sequence[int]): Input feature index. Default: 0
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            Default: None.

            - 'resize_concat': Multiple feature maps will be resized to the
                same size as the first one and then concat together.
                Usually used in FCN head of HRNet.
            - 'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            - None: Only one select feature map is allowed.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        loss_keypoint (dict): Config for keypoint loss. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 extra=None,
                 in_index=0,
                 input_transform=None,
                 align_corners=False,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None,
                 upsample=0,):
        super().__init__()

        self.in_channels = in_channels
        self.loss = build_loss(loss_keypoint)
        self.upsample = upsample

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')

        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = in_index
        self.align_corners = align_corners

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_deconv_layers ({num_deconv_layers}) should >= 0.')

        # proto head after deconv
        in_channels = num_deconv_filters[-1]
        channels = extra['channels']
        act = extra['activation']
        act_fn = nn.ReLU if act == 'relu' else nn.SiLU
        neck_type = extra['neck_type']

        if neck_type == 'conv':
            self.neck = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(channels),
                act_fn(inplace=True),
                )
        elif neck_type == 'res2':
            layers = []
            for bi in range(2):
                if in_channels != channels:
                    downsample = nn.Sequential(
                                    nn.Conv2d(in_channels, channels, 1, bias=False),
                                    nn.BatchNorm2d(channels))
                else:
                    downsample = None
                layers.append(ResBlock(in_channels, channels,
                                    downsample=downsample, act=act))
                in_channels = channels
            self.neck = nn.Sequential(*layers)
        else:
            if in_channels != channels:
                downsample = nn.Sequential(
                                nn.Conv2d(in_channels, channels, 1, bias=False),
                                nn.BatchNorm2d(channels))
            else:
                downsample = None
            self.neck = ResBlock(in_channels, channels,
                                downsample=downsample, act=act)
        
        head_kernel = extra['head_kernel']
        if head_kernel == 3:
            self.head = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.BatchNorm2d(channels),
                act_fn(inplace=True),
                nn.Conv2d(channels, out_channels, 1))
        else:
            self.head = nn.Sequential(
                nn.Conv2d(channels, channels, 1),
                nn.BatchNorm2d(channels),
                act_fn(inplace=True),
                nn.Conv2d(channels, out_channels, 1))

        self.num_keypoints = extra['num_keypoints']
        self.kpt_idxs = []
        start_k = 0
        for k in self.num_keypoints:
            self.kpt_idxs.append(list(range(start_k,start_k+k)))
            start_k += k

        #kpt_proto_cfg = {'cfg':extra, 'out_channels':out_channels, 'num_keypoints': self.num_keypoints}
        self.kpt_prototype = KptPrototype(extra, out_channels, self.num_keypoints)

        # keypoint conditional modules
        self.is_kpt_cond = False
        if 'kpt_cond' in extra.keys():
            kpt_cond = extra['kpt_cond']
            is_kpt_cond = kpt_cond['enabled']
            self.is_kpt_cond = is_kpt_cond

            if self.is_kpt_cond:
                self.enc_layer = nn.Conv2d(1, out_channels, 3, 1, 1)

                self.fusion_method = kpt_cond['fusion_method']
                if self.fusion_method == 'sum':
                    self.fusion_layer = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                        nn.BatchNorm2d(out_channels),
                        act_fn(inplace=True),
                        nn.Conv2d(out_channels, out_channels, 1))
                else:
                    self.fusion_layer = nn.Sequential(
                        nn.Conv2d(out_channels*2, out_channels, 3, 1, 1),
                        nn.BatchNorm2d(out_channels),
                        act_fn(inplace=True),
                        nn.Conv2d(out_channels, out_channels, 1))
                drop_path = 0.1
                self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if dist.is_available() and dist.is_initialized(): # does this work in multi-node setup?
            self.ngpus = dist.get_world_size()
            #print(f"self.ngpus: {self.ngpus}")

        self.cluster_conf_thr = 0.0
        if 'cluster_conf_thr' in extra.keys():
            self.cluster_conf_thr = extra['cluster_conf_thr']
        self.phm_loss_weight = extra['phm_loss_weight']
        self.css_enabled = False
        self.css_weight = extra['css_weight']
        self.css_conf_thr = extra['css_conf_thr']
        self.css_update_weight = extra['css_update_weight']
        self.css_match_dist = extra.get('css_match_dist', 2.1)
        self.css_target_generator = TopDownGenerateTarget(sigma=2,
                                                    kernel=(11,11),
                                                    target_type='GaussianHeatmap',
                                                    encoding='UDP')

        self.cps_weight = extra.get('cps_weight', 0)
        self.cps_enabled = False
        cp_size = extra.get('cp_size', 0) # set the correct value if you resume from cps-enabled model
        if cp_size > 0:
            self.cluster_prototype = KptPrototype(extra, out_channels, [cp_size]) # this is just dummy
            self.cluster_prototype.freeze_weight = True
        self.extra = extra
        #self.assigns = None
        #self.kpt_to_cst = None
        
        self.proto_loss = build_loss(extra['loss_proto'])
        self.is_freeze_weight = False

    def freeze_weight(self):
        #print("Freezing prototype weights")
        self.is_freeze_weight = True
        self.kpt_prototype.freeze_weight = True
    
    def thaw_weight(self):
        #print("Thawing prototype weights")
        self.is_freeze_weight = False
        self.kpt_prototype.freeze_weight = False

    def set_cluster_prototypes(self, assigns, closest_proto_idxs, centroids):
        """Initialize cluster prototypes.
        Note:
            - number of total keypoints: J
            - number of clusters: C
            - number of in-cluster prototypes: M
            - embedding dimension: D

        Args:
            assigns [J, C]: soft assignments for the keypoint prototypes
            closest_proto_idxs [C]: indices of the cumulative keypoints closest to each centroid
            centroids [C, M, D]: centroid features
        """
        J, C = assigns.shape
        C, M, D = centroids.shape

        extra = self.extra
        self.cps_enabled = True
        self.cluster_prototype = KptPrototype(extra, D, [C])
        self.cluster_prototype.prototypes = nn.Parameter(centroids, requires_grad=False)
        self.cluster_prototype.freeze_weight = True
        #self.assigns = assigns
        #self.kpt_to_cst = torch.argmax(assigns, dim=1)

        num_kpts_cumsum = [0] + [sum(self.num_keypoints[:i+1]) for i in range(len(self.num_keypoints))]

        self.kpt_conv_idxs = []
        self.cst_conv_idxs = []
        for d_i, n_kpt_sum in enumerate(num_kpts_cumsum):
            if d_i == 0:
                continue
            d_i_start_kpt_id = num_kpts_cumsum[d_i-1]
            d_i_cluster_mask = (closest_proto_idxs < n_kpt_sum) * (closest_proto_idxs >= d_i_start_kpt_id)
            d_i_cluster_idxs = d_i_cluster_mask.nonzero().squeeze(1)
            d_i_kpt_idxs = closest_proto_idxs[d_i_cluster_mask] - d_i_start_kpt_id
            self.kpt_conv_idxs.append(d_i_kpt_idxs)
            self.cst_conv_idxs.append(d_i_cluster_idxs)
        
        print("Initialized cluster prototypes")


    def hm_sample(self, emb, hm, thr):
        """
        emb: [N, C, H, W]
        hm: [N, K, H, W]
        thr: scalar value.
        Sample embeddings based on the given hm.
        """
        N, C, H, W = emb.shape
        K = hm.size(1)

        emb_flat = emb.permute(0,2,3,1).reshape(-1,C)
        hm_flat = hm.permute(0,2,3,1).reshape(-1,K)

        hm_max, hm_idxs = hm_flat.max(dim=1)
        valid_idxs = hm_max>thr
        valid_classes = hm_idxs[valid_idxs]
        emb_params = emb_flat[valid_idxs]
        valid_conf = hm_max[valid_idxs]

        return emb_params, valid_classes, valid_conf
    
    def get_prt_loss(self, output, pred_hm, targets, target_weights, dataset_ids):
        """ get prototype loss
        Args:
            output [B, D, H, W]: embedding map
            pred_hm [B, K, H, W]: predicted heatmap from the embedding map
            targets [D x [B, K_d, H, W]]: list of target keypoint heatmaps
            target_weights [D x [B, K_d, 1]]: list of target keypoint weights
            dataset_ids [B]: dataset ids
        """
        B, D, H, W = output.shape
        num_datasets = len(self.num_keypoints)

        proto_loss = torch.zeros(1, dtype=output.dtype, device=output.device).sum()
        hm_loss = torch.zeros(1, dtype=output.dtype, device=output.device).sum()

        for di in range(num_datasets):
            di_mask = dataset_ids==di
            di_num = di_mask.sum().item()
            if di_num == 0:
                continue
            di_kpt_idxs = self.kpt_idxs[di]
            num_kpts = self.num_keypoints[di]

            emb_di = output[di_mask]
            gt_hm_di = targets[di][di_mask]
            gt_w_di = target_weights[di][di_mask]
            sample_mask = gt_hm_di

            # kpt emb sampling
            sample_embs, sample_classes, sample_confs = self.hm_sample(emb_di, sample_mask, self.cluster_conf_thr)

            if len(sample_embs) == 0:
                continue
            
            proto_dict = self.kpt_prototype(sample_embs, sample_classes, sample_confs, di)
            proto_logits = proto_dict['logits']
            proto_targets = proto_dict['target']

            _proto_loss, _proto_loss_dict = self.proto_loss(proto_logits, proto_targets, num_kpts)
            proto_loss = proto_loss + _proto_loss

            # hm loss
            pred_hm_di = pred_hm[di_mask]
            pred_hm_di_sub = pred_hm_di[:, di_kpt_idxs]
            hm_loss = hm_loss + self.loss(pred_hm_di_sub, gt_hm_di, gt_w_di)
        return proto_loss, hm_loss

    def get_cps_loss(self, output, targets, target_weights, dataset_ids):
        """ get clustered prototypes supervision loss
        Args:
            output [B, D, H, W]: embedding map
            targets [D x [B, K_d, H, W]]: list of target keypoint heatmaps
            target_weights [D x [B, K_d, 1]]: list of target keypoint weights
            dataset_ids [B]: dataset ids
        """
        B, D, H, W = output.shape

        # proto head's predicted hm
        emb_flat = rearrange(output, 'b c h w -> (b h w) c')
        pred_hm_flat = self.cluster_prototype(emb_flat, dataset_idx=-1, infer_only=True)['kpt_class']
        pred_hm = rearrange(pred_hm_flat, '(b h w) k -> b k h w ', b=B, h=H, w=W)
        C = pred_hm.size(1)
        
        # prepare cluster targets
        total_embs, total_clusters, total_confs = [], [], []
        cst_targets, cst_target_weights = [], []
        device = targets[0].device
        num_datasets = len(self.num_keypoints)
        for di in range(num_datasets):
            di_mask = dataset_ids==di
            di_num = di_mask.sum().item()
            if di_num == 0:
                continue
            
            kpt_conv_idxs = self.kpt_conv_idxs[di]
            cst_conv_idxs = self.cst_conv_idxs[di]

            emb_di = output[di_mask]
            gt_hm_di = targets[di][di_mask]
            gt_w_di = target_weights[di][di_mask]
            
            cst_targets_di = torch.zeros((di_num, C, H, W), dtype=targets[0].dtype, device=device)
            cst_target_weights_di = torch.zeros((di_num, C, 1), dtype=target_weights[0].dtype, device=device)

            cst_targets_di[:, cst_conv_idxs] = gt_hm_di[:, kpt_conv_idxs]
            cst_target_weights_di[:, cst_conv_idxs] = gt_w_di[:, kpt_conv_idxs]
            
            cst_targets.append(cst_targets_di)
            cst_target_weights.append(cst_target_weights_di)

            # cluster emb sampling
            sample_mask = cst_targets_di
            sample_embs, sample_classes, sample_confs = self.hm_sample(emb_di, sample_mask, self.cluster_conf_thr)

            if len(sample_embs) == 0:
                continue
            
            total_embs.append(sample_embs)
            total_clusters.append(sample_classes)
            total_confs.append(sample_confs)
                    
        # concat
        total_embs = torch.cat(total_embs, dim=0)
        total_clusters = torch.cat(total_clusters, dim=0)
        total_confs = torch.cat(total_confs, dim=0)
        cst_targets = torch.cat(cst_targets, dim=0)
        cst_target_weights = torch.cat(cst_target_weights, dim=0)

        # proto loss
        proto_dict = self.cluster_prototype(total_embs, total_clusters, total_confs, 0)
        proto_logits = proto_dict['logits']
        proto_targets = proto_dict['target']

        proto_loss, proto_loss_dict = self.proto_loss(proto_logits, proto_targets, C)

        # hm loss
        hm_loss = self.loss(pred_hm, cst_targets, cst_target_weights)

        cps_loss = proto_loss + hm_loss * self.phm_loss_weight
        cps_loss = cps_loss * self.cps_weight

        return cps_loss

    def get_css_preds(self, heatmaps, kernel=11):
        """Get keypoint predictions from the heatmaps
        Note: We actually implemented pytorch version of this, but turned out that numpy is empirically faster.
        """
        heatmaps = heatmaps.detach().cpu().numpy()
        preds, maxvals = _get_max_preds(heatmaps)
        preds = post_dark_udp(preds, heatmaps, kernel=kernel)
        return preds, maxvals

    def get_css_loss(self, output, targets, target_weights, dataset_ids, pred_hm, pred_hms):
        """Calculate cross-dataset self-supervision loss
        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            targets D x (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weights D x (torch.Tensor[N,K,1]):
                Weights across different joint types.
            dataset_ids (torch.Tensor[N,]): dataset ids.
            pred_hm D x (torch.Tensor[N,K,H,W]): Predicted heatmaps from the prototype head.
            pred_hms D x (torch.Tensor[N,K,H,W]): Predicted heatmaps from the keypoint heads.
        """
        dtype = torch.float32
        device = targets[0].device
        
        css_loss = .0

        num_datasets = len(self.num_keypoints)
        for di in range(num_datasets):
            neg_di_mask = dataset_ids!=di
            neg_di_num = neg_di_mask.sum().item()
            if neg_di_num == 0:
                continue
            di_kpt_idxs = self.kpt_idxs[di]

            mh_hm_di = pred_hms[di][neg_di_mask]
            prt_hm_di = pred_hm[neg_di_mask][:, di_kpt_idxs]

            Nd, Kd, H, W = mh_hm_di.shape

            # apply initial filtering before running numpy APIs
            mh_hm_di_v = mh_hm_di.amax(dim=-1).amax(dim=-1) > self.css_conf_thr
            prt_hm_di_v = prt_hm_di.amax(dim=-1).amax(dim=-1) > self.css_conf_thr
            hm_di_v = mh_hm_di_v * prt_hm_di_v # filter by confidence score
            n_hm_di_v = hm_di_v.sum().item()
            if n_hm_di_v == 0:
                continue
                        
            mh_hm_di_max_x, mh_hm_di_max_y = torch.argmax(mh_hm_di.amax(2), dim=2), torch.argmax(mh_hm_di.amax(3), dim=2)
            prt_hm_di_max_x, prt_hm_di_max_y = torch.argmax(prt_hm_di.amax(2), dim=2), torch.argmax(prt_hm_di.amax(3), dim=2)
            max_idx_v = torch.sqrt((mh_hm_di_max_x - prt_hm_di_max_x) ** 2 + (prt_hm_di_max_x - prt_hm_di_max_y) ** 2) <= self.css_match_dist

            filter_mask = hm_di_v * max_idx_v

            inst_filter_mask = filter_mask.sum(dim=1)>0
            n_inst_filter = inst_filter_mask.sum().item()
            if n_inst_filter == 0:
                continue
            inst_joint_filter_mask = filter_mask[inst_filter_mask]

            mh_hm_di = mh_hm_di[inst_filter_mask]
            prt_hm_di = prt_hm_di[inst_filter_mask]
            emb_di = output[neg_di_mask][inst_filter_mask]
            mh_hm_di_filt = mh_hm_di[inst_joint_filter_mask] # [Nf, H, W]
            prt_hm_di_filt = prt_hm_di[inst_joint_filter_mask] # [Nf, H, W]
            
            mh_preds, mh_maxvals = self.get_css_preds(mh_hm_di_filt.unsqueeze(0))
            prt_preds, prt_maxvals = self.get_css_preds(prt_hm_di_filt.unsqueeze(0))
            mh_preds, mh_maxvals = mh_preds[0], mh_maxvals[0]
            prt_preds, prt_maxvals = prt_preds[0], prt_maxvals[0]
            Nf = mh_preds.shape[0]

            # get pseudo gt kpt
            maxvals_sum = np.clip(mh_maxvals + prt_maxvals, 1.0e-4, None)
            maxvals_min = np.min(np.stack((mh_maxvals, prt_maxvals)), axis=0)
            mh_maxvals_rescale = mh_maxvals / maxvals_sum
            prt_maxvals_rescale = prt_maxvals / maxvals_sum
            pseudo_gt_kpts = mh_maxvals_rescale * mh_preds + prt_maxvals_rescale * prt_maxvals
            all_valid = (mh_maxvals > self.css_conf_thr) * (prt_maxvals > self.css_conf_thr) * 1.0
            pseudo_gt_kpts = np.concatenate((pseudo_gt_kpts, all_valid), axis=1)

            # make pseudo gt hm
            factors = self.css_target_generator.sigma
            gen_cfg = {
                'num_joints': Nf,
                'image_size': np.array([W, H]),
                'heatmap_size': np.array([W, H]),
                'joint_weights': [1.0] * Nf, # we don't use this anyway
                'use_different_joint_weights': False,
            }
            """
            _css_target, _css_target_weight = self.css_target_generator._msra_generate_target(
                                    gen_cfg,
                                    pseudo_gt_kpts, all_valid, self.css_target_generator.sigma)
            """
            _css_target, _css_target_weight = self.css_target_generator._udp_generate_target(
                            gen_cfg, 
                            pseudo_gt_kpts, all_valid, factors,
                            self.css_target_generator.target_type)
            
            _css_target = torch.from_numpy(_css_target).to(dtype).to(device)
            _css_target_weight = torch.from_numpy(_css_target_weight).to(dtype).to(device)

            # assign back to 4-dim tensor
            css_target = torch.zeros((n_inst_filter, Kd, H, W), dtype=dtype, device=device)
            css_target_weight = torch.zeros((n_inst_filter, Kd, 1), dtype=dtype, device=device)
            css_target[inst_joint_filter_mask] = _css_target
            #css_target_weight[inst_joint_filter_mask] = _css_target_weight
            css_target_weight[inst_joint_filter_mask] = torch.from_numpy(maxvals_min).to(dtype).to(device)

            hm_loss = self.loss(mh_hm_di, css_target, css_target_weight)

            phm_loss = self.loss(prt_hm_di, css_target, css_target_weight) * self.phm_loss_weight

            sample_mask = css_target * css_target_weight[:,:,:,None]
            sample_embs, sample_classes, sample_confs = self.hm_sample(emb_di, sample_mask, self.cluster_conf_thr)
            if len(sample_embs) == 0:
                continue
            proto_dict = self.kpt_prototype(sample_embs, sample_classes, sample_confs, di)
            proto_logits = proto_dict['logits']
            proto_targets = proto_dict['target']
            _proto_loss, _proto_loss_dict = self.proto_loss(proto_logits, proto_targets, Kd)

            if torch.isnan(hm_loss).any().item():
                print("Nan value in hm_loss of css")
                print(f"mh_hm_di.min(): {mh_hm_di.amin()}, mh_hm_di.max(): {mh_hm_di.amax()}")
                print(f"css_target.min(): {css_target.amin()}, css_target.max(): {css_target.amax()}")
                print(f"css_target_weight.min(): {css_target_weight.amin()}, css_target_weight.max(): {css_target_weight.amax()}")
                exit()
            
            if torch.isnan(phm_loss).any().item():
                print("Nan value in phm_loss of css")
                print(f"prt_hm_di.min(): {prt_hm_di.amin()}, prt_hm_di.max(): {prt_hm_di.amax()}")
                print(f"css_target.min(): {css_target.amin()}, css_target.max(): {css_target.amax()}")
                print(f"css_target_weight.min(): {css_target_weight.amin()}, features.max(): {css_target_weight.amax()}")
                exit()

            if torch.isnan(_proto_loss).any().item():
                print("Nan value in _proto_loss of css")
                print(f"emb_di.min(): {emb_di.amin()}, emb_di.max(): {emb_di.amax()}")
                print(f"proto_logits.min(): {proto_logits.amin()}, proto_logits.max(): {proto_logits.amax()}")
                print(f"proto_targets.min(): {proto_targets.amin()}, proto_targets.max(): {proto_targets.amax()}")
                exit()
            
            css_loss = css_loss + hm_loss + phm_loss + _proto_loss

        return css_loss


    def get_loss(self, output, targets, target_weights, pred_hms, dataset_ids):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target D x (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight D x (torch.Tensor[N,K,1]):
                Weights across different joint types.
            pred_hms D x (torch.Tensor[N,K,H,W]): Predicted heatmaps.
            dataset_ids (torch.Tensor[N,]): dataset ids.
        """

        losses = dict()

        assert not isinstance(self.loss, nn.Sequential)
        assert targets[0].dim() == 4 and target_weights[0].dim() == 3

        B = output.size(0)
        H, W = output.shape[2:]

        # proto head's predicted hm
        emb_flat = rearrange(output, 'b c h w -> (b h w) c')
        pred_hm_flat = self.kpt_prototype(emb_flat, dataset_idx=-1, infer_only=True)['kpt_class']
        pred_hm = rearrange(pred_hm_flat, '(b h w) k -> b k h w ', b=B, h=H, w=W)

        proto_loss, hm_loss = self.get_prt_loss(output, pred_hm, targets, target_weights, dataset_ids)
        """
        proto_loss = torch.zeros(1, dtype=output.dtype, device=output.device).sum()
        hm_loss = torch.zeros(1, dtype=output.dtype, device=output.device).sum()
        num_datasets = len(self.num_keypoints)
        for di in range(num_datasets):
            di_mask = dataset_ids==di
            di_num = di_mask.sum().item()
            if di_num == 0:
                continue
            di_kpt_idxs = self.kpt_idxs[di]
            num_kpts = self.num_keypoints[di]

            emb_di = output[di_mask]
            gt_hm_di = targets[di][di_mask]
            gt_w_di = target_weights[di][di_mask]
            sample_mask = gt_hm_di

            # kpt emb sampling
            sample_embs, sample_classes, sample_confs = self.hm_sample(emb_di, sample_mask, self.cluster_conf_thr)

            if len(sample_embs) == 0:
                continue
            
            proto_dict = self.kpt_prototype(sample_embs, sample_classes, sample_confs, di)
            proto_logits = proto_dict['logits']
            proto_targets = proto_dict['target']

            _proto_loss, _proto_loss_dict = self.proto_loss(proto_logits, proto_targets, num_kpts)
            proto_loss = proto_loss + _proto_loss

            # hm loss
            pred_hm_di = pred_hm[di_mask]
            pred_hm_di_sub = pred_hm_di[:, di_kpt_idxs]
            hm_loss = hm_loss + self.loss(pred_hm_di_sub, gt_hm_di, gt_w_di)
        """
        losses['proto_loss'] = proto_loss # weight is computed inside loss function
        losses['phm_loss'] = hm_loss * self.phm_loss_weight

        if self.cps_enabled:
            cps_loss = self.get_cps_loss(output, targets, target_weights, dataset_ids)
            losses['cps_loss'] = cps_loss
            if torch.isnan(cps_loss).any().item():
                print("Nan value in cps_loss")
                print(f"out_emb.min(): {out_emb.min()}, out_emb.max(): {out_emb.max()}")
                print(f"neck_feat.min(): {neck_feat.min()}, neck_feat.max(): {neck_feat.max()}")
                print(f"features.min(): {features.min()}, features.max(): {features.max()}")
                exit()

        # css
        if self.css_weight > 0 and self.css_enabled:
            css_loss = self.get_css_loss(output, targets, target_weights, dataset_ids, pred_hm, pred_hms)
            losses['css_loss'] = css_loss * self.css_weight

        if torch.isnan(proto_loss).any().item():
            print("Nan value in proto_loss")
            print(f"ppc: {ppc}, ppd: {ppd}")
            print(f"out_emb.min(): {out_emb.min()}, out_emb.max(): {out_emb.max()}")
            print(f"neck_feat.min(): {neck_feat.min()}, neck_feat.max(): {neck_feat.max()}")
            print(f"features.min(): {features.min()}, features.max(): {features.max()}")
            exit()
        
        if dist.is_available() and dist.is_initialized() and not self.is_freeze_weight:
            protos = self.kpt_prototype.prototypes.data.clone()
            dist.all_reduce(protos.div_(self.ngpus))
            self.kpt_prototype.prototypes = nn.Parameter(protos, requires_grad=False)

        return losses

    # this should not be called.
    def get_accuracy(self, output, target, target_weight, ):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """

        accuracy = dict()

        if self.target_type == 'GaussianHeatmap':
            _, avg_acc, _ = pose_pck_accuracy(
                output.detach().cpu().numpy(),
                target.detach().cpu().numpy(),
                target_weight.detach().cpu().numpy().squeeze(-1) > 0)
            accuracy['acc_pose'] = float(avg_acc)

        return accuracy

    def forward(self, x, k=None, dataset_source=None):
        """Forward function."""
        x = self._transform_inputs(x)
        x = self.deconv_layers(x)
        x = self.neck(x)
        x = self.head(x)

        if self.is_kpt_cond and k is not None:
            if isinstance(k, list):
                k = torch.cat(k, dim=1)
            k_max = k.amax(dim=1, keepdim=True)
            enc = self.enc_layer(k_max)

            if self.fusion_method == 'sum':
                x = self.fusion_layer(x + self.drop_path(enc))
            else:
                x_cat = torch.cat([x, self.drop_path(enc)], dim=1)
                x = self.fusion_layer(x_cat)
  
        return x

    def inference_model(self, x, flip_pairs=None, target_dataset=-1):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (torch.Tensor[N,K,H,W]): Input features.
            flip_pairs (None | list[tuple]):
                Pairs of keypoints which are mirrored.
            target_dataset (int): dataset index. -1: outputs all dataset keypoints
        """
        output = self.forward(x)

        b, c, h, w = output.shape
        sample_embs = rearrange(output, 'b c h w -> (b h w) c')
        proto_dict = self.kpt_prototype(sample_embs, None, None, target_dataset)
        proto_hm = proto_dict['kpt_class']
        proto_hm = rearrange(proto_hm, '(b h w) k -> b k h w', b=b, h=h, w=w)
        
        if flip_pairs is not None:
            output_heatmap = flip_back(
                proto_hm.detach().cpu().numpy(),
                flip_pairs,
                target_type=self.target_type)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg.get('shift_heatmap', False):
                output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        else:
            output_heatmap = proto_hm.detach().cpu().numpy()
        
        if not self.cps_enabled:
            return output_heatmap

        cp_dict = self.cluster_prototype(sample_embs, None, None, -1)
        cp_hm = cp_dict['kpt_class']
        cp_hm = rearrange(cp_hm, '(b h w) k -> b k h w', b=b, h=h, w=w)

        """
        if flip_pairs is not None:
            output_cphm = flip_back(
                cp_hm.detach().cpu().numpy(),
                flip_pairs,
                target_type=self.target_type)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg.get('shift_heatmap', False):
                output_cphm[:, :, :, 1:] = output_cphm[:, :, :, :-1]
        else:
            output_cphm = cp_hm.detach().cpu().numpy()
        """
        output_cphm = cp_hm.detach().cpu().numpy() # no flip possible
        return output_heatmap, output_cphm
        

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform is not None, in_channels and in_index must be
        list or tuple, with the same length.

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.

                - 'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                - 'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                - None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list):
            if not isinstance(inputs, list):
                if self.upsample > 0:
                    inputs = resize(
                        input=F.relu(inputs),
                        scale_factor=self.upsample,
                        mode='bilinear',
                        align_corners=self.align_corners
                        )
            return inputs

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    def init_weights(self):
        """Initialize model weights."""
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)


