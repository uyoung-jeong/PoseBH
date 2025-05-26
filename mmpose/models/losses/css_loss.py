import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from ..builder import LOSSES

from mmpose.models.builder import build_loss
from mmpose.datasets.pipelines.top_down_transform import TopDownGenerateTarget
from mmpose.core.evaluation.top_down_eval import keypoints_from_heatmaps

@LOSSES.register_module()
class CSSLoss(nn.Module):
    """
    Continual Self Supervision Loss
    """
    def __init__(self, cfg=None):
        super(CSSLoss, self).__init__()
        self.cfg = cfg
        self.weight = cfg['weight']
        self.css_conf_thr = cfg['css_conf_thr']
        self.target_generator = TopDownGenerateTarget(sigma=2,
                                                        kernel=(11,11),
                                                        target_type='GaussianHeatmap',
                                                        encoding='UDP')
        self.hm_loss = build_loss(dict(type='JointsMSELoss', use_target_weight=True))
        
    # preds: [n, K, H, W]
    # targets: [n, H, H, W]
    def forward(self, preds, targets):
        # get keypoint predictions from the keypoint head
        N, K, H, W = targets.shape
        center = np.repeat(np.array([(W-1)*0.5, (H-1)*0.5]).reshape(1,2), N, axis=0)
        scale = np.repeat(np.array([W-1, H-1]).reshape(1,2), N, axis=0) / 200.0
        pred_kpts, pred_maxvals = keypoints_from_heatmaps(
            preds.detach().cpu().numpy(),
            center,
            scale,
            unbiased=False,
            post_process='default',
            kernel=11,
            valid_radius_factor=0.0546875,
            use_udp=True,
            target_type='GaussianHeatmap')
        
        # filter out low-confidence keypoints
        pred_kpts_flat = pred_kpts[:,:,:2].reshape(N*K, 2)
        pred_maxvals_flat = pred_maxvals.reshape(-1,1)
        valid_mask = (pred_maxvals_flat > self.css_conf_thr).flatten()
        n_valid = valid_mask.sum().item()
        if valid_mask == 0:
            return 0.0
        
        pred_kpts_v = pred_kpts_flat[valid_mask]
        pred_maxvals_v = pred_maxvals_flat[valid_mask]
        Nf = pred_kpts_v.shape[0]
        preds_v = preds.reshape(N*K,H,W)[valid_mask]
        
        # make reliable heatmap
        factors = self.target_generator.sigma
        channel_factor = 1
        udp_cfg = {
            'num_joints': Nf,
            'image_size': np.array([W, H]),
            'heatmap_size': np.array([W, H]),
            'joint_weights': [1.0] * Nf, # we don't use this anyway
            'use_different_joint_weights': False,
        }
        css_target, css_target_weight = self.target_generator._udp_generate_target(
                udp_cfg, pred_kpts_v, pred_maxvals_v, factors,
                self.target_generator.target_type)
        
        # get loss
        dtype = torch.float32
        device = preds.device
        css_target = torch.from_numpy(css_target).to(dtype).to(device)
        css_target_weight = torch.from_numpy(css_target_weight).to(dtype).to(device)
        css_target_weight = css_target_weight.unsqueeze(2)

        loss = self.hm_loss(preds_v[None,:], css_target[None,:], css_target_weight[None,:]) * self.weight
        return loss
