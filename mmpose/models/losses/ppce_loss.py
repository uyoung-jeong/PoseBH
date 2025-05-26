import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

from ..builder import LOSSES

class PPC(nn.Module):
    def __init__(self, cfg):
        super(PPC, self).__init__()
        self.cfg = cfg
        self.ignore_label = -1
        self.bg_reweight = True # cfg.PROTO.BG_REWEIGHT
        self.num_in_cls_proto = cfg['num_in_class_proto']
        self.num_joints = cfg['num_joints']
        self.learn_bg = False # cfg.PROTO.LEARN_BG

    def forward(self, contrast_logits, contrast_target, num_joints=-1):
        valid_target = contrast_target[contrast_target != self.ignore_label]
        if num_joints > 0:
            max_cls = num_joints * self.num_in_cls_proto
        else:
            max_cls = self.num_joints * self.num_in_cls_proto
        weight_size = max_cls+self.num_in_cls_proto if self.learn_bg else max_cls
        weight = torch.ones(weight_size,dtype=torch.float32, device=valid_target.device)
        if self.bg_reweight and self.learn_bg:
            num_bg = torch.sum(valid_target>=max_cls).item()
            bg_weight = self.num_in_cls_proto * (valid_target.size(0) - num_bg)/(1 + num_bg * (max_cls + self.num_in_cls_proto)) # total_num / (bg_num * cls_size)
            bg_weight = min(bg_weight, 1.0)
            weight[-1] = bg_weight
        loss_ppc = F.cross_entropy(contrast_logits, contrast_target.long(), weight=weight, ignore_index=self.ignore_label)
        return loss_ppc


class PPD(nn.Module):
    def __init__(self, cfg):
        super(PPD, self).__init__()
        self.cfg = cfg
        self.ignore_label = -1
        self.bg_reweight = True #cfg.PROTO.BG_REWEIGHT
        self.num_in_cls_proto = cfg['num_in_class_proto']
        self.num_joints = cfg['num_joints']
        self.learn_bg = False #cfg.PROTO.LEARN_BG

    def forward(self, contrast_logits, contrast_target, num_joints=-1):
        valid_target = contrast_target != self.ignore_label
        contrast_logits = contrast_logits[valid_target, :]
        contrast_target = contrast_target[valid_target]

        logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
        loss_ppd = (1 - logits).pow(2)

        if self.bg_reweight and self.learn_bg:
            if num_joints > 0:
                max_cls = num_joints * self.num_in_cls_proto
            else:
                max_cls = self.num_joints * self.num_in_cls_proto
            bg_idxs = contrast_target>=max_cls
            num_bg = torch.sum(bg_idxs).item()
            bg_weight = self.num_in_cls_proto * (contrast_target.size(0) - num_bg)/(1 + num_bg * (max_cls+self.num_in_cls_proto)) # total_num / bg_num
            bg_weight = min(bg_weight, 1.0)
            weights = torch.ones_like(contrast_target).to(torch.float32)
            weights[bg_idxs] = bg_weight
            loss_ppd = loss_ppd.reshape(-1) * weights
        
        loss_ppd = loss_ppd.mean()

        return loss_ppd

@LOSSES.register_module()
class PixelPrototypeCELoss(nn.Module):
    def __init__(self, cfg=None):
        super(PixelPrototypeCELoss, self).__init__()
        self.cfg = cfg
        self.weight = cfg['weight']
        self.ppc_weight = cfg['ppc_weight']
        self.ppd_weight = cfg['ppd_weight']
        num_joints = cfg['num_joints']
        self.ppc_criterion = PPC(cfg)
        self.ppd_criterion = PPD(cfg)

    # preds: [n, Kxm]
    # target: [n]
    # classes: [n]
    def forward(self, preds, target, num_joints=-1):
        loss_ppc = self.ppc_criterion(preds, target, num_joints) * self.ppc_weight
        loss_ppd = self.ppd_criterion(preds, target, num_joints) * self.ppd_weight
        loss = loss_ppc + loss_ppd
        loss_dict = {'ppc': loss_ppc.detach(), 'ppd': loss_ppd.detach()}
        loss = loss * self.weight
        return loss, loss_dict
