import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from collections import defaultdict
import math
from einops import rearrange, repeat


import copy
from itertools import combinations
import numpy as np

from ..builder import HEADS

# https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/layers/weight_init.py
def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor

# https://github.com/tfzhou/ProtoSeg/blob/main/lib/models/modules/sinkhorn.py#L5
def distributed_sinkhorn(out, sinkhorn_iterations=3, epsilon=0.05, hard=True):
    L = torch.exp(out / epsilon).t() # K x B
    B = L.shape[1]
    K = L.shape[0]

    # make the matrix sums to 1
    sum_L = torch.sum(L)
    L /= sum_L

    for _ in range(sinkhorn_iterations):
        L /= torch.sum(L, dim=1, keepdim=True)
        L /= K

        L /= torch.sum(L, dim=0, keepdim=True)
        L /= B

    L *= B
    L = L.t()

    indexs = torch.argmax(L, dim=1)
    # L = torch.nn.functional.one_hot(indexs, num_classes=L.shape[1]).float()
    L = F.gumbel_softmax(L, tau=0.5, hard=hard)

    return L, indexs

# https://github.com/tfzhou/ProtoSeg/blob/main/lib/models/modules/contrast.py
def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
            momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
            torch.norm(update, p=2)))
    return update

# https://github.com/tfzhou/ProtoSeg/blob/main/lib/models/nets/hrnet.py
# prototype class
@HEADS.register_module()
class KptPrototype(nn.Module):
    def __init__(self, cfg, out_channels, num_keypoints):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.total_num_keypoints = sum(num_keypoints)
        self.num_datasets = len(num_keypoints)
        self.num_in_cls_proto = cfg['num_in_class_proto']
        self.init_qr = False
        
        self.emb_dim = out_channels
        self.learn_bg = False
        self.num_proto_cls = self.total_num_keypoints+1 if self.learn_bg else self.total_num_keypoints
        self.init_proto()
        self.gamma = cfg['gamma']
        self.update_prototype = True
        self.cluster_conf_thr = cfg.get('cluster_conf_thr', 0.0) # 0.0 by default for soft clustering
        self.is_hard_cluster = cfg.get('hard_cluster', False) # False by default for soft clustering

        if dist.is_available() and dist.is_initialized():
            self.ngpus = dist.get_world_size()

        self.kpt_idxs = []
        start_k = 0
        for k in self.num_keypoints:
            self.kpt_idxs.append(list(range(start_k,start_k+k)))
            start_k += k
        
        self.sim_rescale = 'abs'
        self.freeze_weight = False
    
    def init_proto(self): # initialize prototype
        proto = torch.zeros((self.num_proto_cls, self.num_in_cls_proto, self.emb_dim), dtype=torch.float32)
        proto = F.normalize(_trunc_normal_(proto, mean=0., std=0.02, a=-2., b=2.), dim=2)
        self.prototypes = nn.Parameter(proto, requires_grad=False)
        #print(f"prototype shape: {self.prototypes.shape}")
    
    # feats: [N, C]
    # out_cls: [N, K] K: kpt_num + bg
    # gt_cls: [N,]
    # conf: [N]
    # soft_mask: [N, M, K]
    def prototype_learning(self, feats, out_cls, gt_cls, conf, soft_mask, dataset_idx, update_weight, update_prototype=True):
        N, K = out_cls.shape
        emb_dim = self.prototypes.shape[-1]
        sim = torch.mm(feats, self.prototypes.view(-1, emb_dim).t()) # [N, KxM]
        #sim = sim / torch.clamp(torch.linalg.vector_norm(sim, dim=-1, keepdim=True), 1.0e-8)
        if self.sim_rescale == 'affine':
            sim_rescale = (sim + 1) * 0.5
        elif self.sim_rescale == 'relu':
            sim_rescale = F.relu(sim, inplace=True) # we should not give loss for orthogonal vectors
        elif self.sim_rescale == 'abs':
            sim_rescale = torch.abs(sim)
        else:
            sim_rescale = sim
        
        pred_seg = torch.argmax(out_cls, dim=1)
        mask = (gt_cls == pred_seg)
        conf[~mask] = 0

        # operate only on labeled keypoint set
        proto_target = torch.zeros((N), dtype=gt_cls.dtype, device=gt_cls.device) -1

        protos = self.prototypes.data.clone()
        di = dataset_idx
        n_kpt = self.num_keypoints[di]
        start_k = sum(self.num_keypoints[:di])
        
        proto_idxs = [start_k+e for e in list(range(n_kpt*self.num_in_cls_proto))] # kpt idx
        if self.learn_bg:
            proto_idxs += list(range(self.total_num_keypoints*self.num_in_cls_proto,(self.total_num_keypoints+1)*self.num_in_cls_proto)) #bg idx
        proto_logits = sim_rescale[:, proto_idxs]
        
        n_range = n_kpt+1 if self.learn_bg else n_kpt
        for k in range(n_range): # num_proto_cls
            init_q = soft_mask[...,k]
            t_mask = gt_cls == k
            init_q = init_q[t_mask, ...]
            if init_q.shape[0] == 0:
                continue
            
            #q, indexs = distributed_sinkhorn(init_q, hard=self.is_hard_cluster) # [n, m], [n]
            q, indexs = distributed_sinkhorn(init_q) # [n, m], [n]

            if self.is_hard_cluster:
                m_k = mask[t_mask] # [n]
            else:
                m_k = conf[t_mask] # soft clustering?

            c_k = feats[t_mask, ...] # [n, C]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_in_cls_proto) # [n, m]

            m_q = q * m_k_tile  # n x self.num_prototype # [n, m]

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1]) # [] # [n, C]

            c_q = c_k * c_k_tile  # n x embedding_dim # [n, C]

            f = m_q.transpose(0, 1) @ c_q  # self.num_in_cls_proto x embedding_dim

            n = torch.sum(m_q, dim=0) # [m]
            valid_n = n != 0
            """
            if self.num_in_cls_proto > 1:
                q, indexs = distributed_sinkhorn(init_q, hard=self.is_hard_cluster) # [n, m], [n]

                if self.is_hard_cluster:
                    m_k = mask[t_mask] # [n]
                else:
                    m_k = conf[t_mask] # soft clustering?

                c_k = feats[t_mask, ...] # [n, C]

                m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_in_cls_proto) # [n, m]

                m_q = q * m_k_tile  # n x self.num_prototype # [n, m]

                c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1]) # [] # [n, C]

                c_q = c_k * c_k_tile  # n x embedding_dim # [n, C]

                f = m_q.transpose(0, 1) @ c_q  # self.num_in_cls_proto x embedding_dim

                n = torch.sum(m_q, dim=0) # [m]
                valid_n = n != 0
            else:
                valid_n = 0
                f = init_q.transpose(0,1) @ feats[t_mask, ...]
                indexs = 0
                n = init_q.sum(dim=0)
            """

            proto_k_idx = start_k+k if k < n_kpt else self.total_num_keypoints
            if torch.sum(n) > 0 and self.update_prototype is True:
                f = F.normalize(f, p=2, dim=-1)
                momentum = self.gamma if update_weight>=1.0 else 1-(1-self.gamma)*update_weight
                new_value = momentum_update(old_value=protos[proto_k_idx, valid_n, :], new_value=f[valid_n, :],
                                            momentum=momentum, debug=False) # [B, 720]
                protos[proto_k_idx, valid_n, :] = new_value

            #proto_target[t_mask] = indexs + (self.num_in_cls_proto * proto_k_idx)
            proto_target[t_mask] = indexs + self.num_in_cls_proto * k # since logits are compact kd+1 set, also set compact label number
        
        if update_prototype and not self.freeze_weight:
            self.prototypes = nn.Parameter(F.normalize(protos, p=2, dim=-1),
                                        requires_grad=False)
            """
            if dist.is_available() and dist.is_initialized():
                protos = self.prototypes.data.clone()
                dist.all_reduce(protos.div_(self.ngpus))
                self.prototypes = nn.Parameter(protos, requires_grad=False)
            """
        return proto_logits, proto_target
            
    # feats: [N, C]
    # gt: [N]
    # conf: [N]
    # dataset_idx: int
    def forward(self, feats, gt=None, conf=None, dataset_idx=-1, infer_only=False, update_weight=1.0):
        B, C = feats.shape
        out_dict = {}
        with torch.cuda.amp.autocast(enabled=False):
            feats = F.normalize(feats.type(torch.float32), p=2, dim=-1)
            self.prototypes.data.copy_(F.normalize(self.prototypes, p=2, dim=-1))

            # k:#joints+bg, m:#in-cls-proto
            sim = torch.einsum('nd,kmd->nmk', feats, self.prototypes) # [N, M, K]
            
            if self.sim_rescale == 'affine':
                sim_rescale = (sim + 1) * 0.5
            elif self.sim_rescale == 'relu':
                sim_rescale = F.relu(sim, inplace=True) # we should not give loss for orthogonal vectors
            elif self.sim_rescale == 'abs': # this is used by default
                sim_rescale = torch.abs(sim)
            else:
                sim_rescale = sim
            
            #sim_rescale = sim_rescale / torch.clamp(torch.linalg.vector_norm(sim_rescale, dim=1, keepdim=True), 1.0e-8)
            out_cls = torch.amax(sim_rescale, dim=1)
            if self.training and not infer_only:
                contrast_logits, contrast_target = self.prototype_learning(feats, out_cls, gt, conf, sim_rescale,
                                                    dataset_idx, update_weight)
                out_dict['logits'] = contrast_logits
                out_dict['target'] = contrast_target

            if dataset_idx != -1: # if dataset is specified, only return the corresponding results
                kpt_idxs = self.kpt_idxs[dataset_idx]
                out_cls = out_cls[:, kpt_idxs]
            out_cls = torch.clamp(out_cls, min=1.0e-4, max=1-1.0e-4)
            out_dict['kpt_class'] = out_cls

        return out_dict

# residual block with silu
class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1, act='silu'):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out
