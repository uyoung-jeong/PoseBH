# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

import mmcv
import numpy as np
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow

from mmpose.core import imshow_bboxes, imshow_keypoints, imshow_bboxes_pyplot, imshow_keypoints_pyplot
from .. import builder
from ..builder import POSENETS
from .base import BasePose
from mmcv_custom.checkpoint import load_checkpoint
from mmpose.utils import get_root_logger

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16

import matplotlib.pyplot as plt

@POSENETS.register_module()
class TopDownProto(BasePose):
    """Top-down pose detectors with prototype.

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (None): Deprecated arguments. Please use
            `loss_keypoint` for heads instead.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 keypoint_head=None,
                 proto_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 multihead_pretrained=None,
                 loss_pose=None):
        super().__init__()
        self.fp16_enabled = False

        self.backbone = builder.build_backbone(backbone)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if keypoint_head is not None:
            keypoint_head['train_cfg'] = train_cfg
            keypoint_head['test_cfg'] = test_cfg

            if 'loss_keypoint' not in keypoint_head and loss_pose is not None:
                warnings.warn(
                    '`loss_pose` for TopDown is deprecated, '
                    'use `loss_keypoint` for heads instead. See '
                    'https://github.com/open-mmlab/mmpose/pull/382'
                    ' for more information.', DeprecationWarning)
                keypoint_head['loss_keypoint'] = loss_pose

            self.keypoint_head = builder.build_head(keypoint_head)

        # prototype head
        if proto_head is not None:
            proto_head['train_cfg'] = train_cfg
            proto_head['test_cfg'] = test_cfg

            if 'loss_keypoint' not in proto_head and loss_pose is not None:
                warnings.warn(
                    '`loss_pose` for TopDown is deprecated, '
                    'use `loss_keypoint` for heads instead. See '
                    'https://github.com/open-mmlab/mmpose/pull/382'
                    ' for more information.', DeprecationWarning)
                proto_head['loss_keypoint'] = loss_pose
                
            self.proto_head = builder.build_head(proto_head)

        self.init_weights(pretrained=pretrained, 
                multihead_pretrained=multihead_pretrained)

    @property
    def with_neck(self):
        """Check if has neck."""
        return hasattr(self, 'neck')

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')

    @property
    def with_proto(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'proto_head')

    def init_weights(self, pretrained=None, multihead_pretrained=None):
        """Weight initialization for model."""
        self.backbone.init_weights(pretrained)
        if self.with_neck:
            self.neck.init_weights()
        if self.with_keypoint:
            self.keypoint_head.init_weights()
        if self.with_proto:
            self.proto_head.init_weights()
        
        if multihead_pretrained is not None:
            if isinstance(multihead_pretrained, str):
                logger = get_root_logger('INFO')
                print("Initializing entire model with multihead_pretrained weights")
                load_checkpoint(self, multihead_pretrained, strict=False, logger=logger, map_location='cpu')
            else:
                raise TypeError('pretrained must be a str or None.'
                            f' But received {type(pretrained)}.')

    def forward(self,
                img,
                target=None,
                target_weight=None,
                img_metas=None,
                return_loss=True,
                return_heatmap=False,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            - batch_size: N
            - num_keypoints: K
            - num_img_channel: C (Default: 3)
            - img height: imgH
            - img width: imgW
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            img (torch.Tensor[NxCximgHximgW]): Input images.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]): Weights across
                different joint types.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.
            return_heatmap (bool) : Option to return heatmap.

        Returns:
            dict|tuple: if `return loss` is true, then return losses. \
                Otherwise, return predicted poses, boxes, image paths \
                and heatmaps.
        """
        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs)

    @auto_fp16(apply_to=('img', ), out_fp32=True)
    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        output = self.backbone(img)
        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            kpt_out = self.keypoint_head(output)
        if self.with_proto:
            proto_out = self.proto_head(output)

        # if return loss
        with torch.cuda.amp.autocast(enabled=False):
            losses = dict()
            if self.with_keypoint:
                keypoint_losses = self.keypoint_head.get_loss(
                    kpt_out, target, target_weight)
                losses.update(keypoint_losses)
                keypoint_accuracy = self.keypoint_head.get_accuracy(
                    kpt_out, target, target_weight)
                losses.update(keypoint_accuracy)

            if self.with_proto:
                img_sources = torch.from_numpy(np.array([ele['dataset_idx'] for ele in img_metas])).to(img.device)
                proto_losses = self.proto_head.get_loss(
                    proto_out, [target], [target_weight], output, img_sources)
                losses['proto_loss'] = proto_losses['proto_loss']
                losses['phm_loss'] = proto_losses['phm_loss']
                if 'css_loss' in proto_losses:
                    losses['css_loss'] = proto_losses['css_loss']
                if 'cps_loss' in proto_losses:
                    losses['cps_loss'] = proto_losses['cps_loss']
        return losses

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]

        result = {}

        features = self.backbone(img)
        if self.with_neck:
            features = self.neck(features)
        if self.with_keypoint:
            output_heatmap = self.keypoint_head.inference_model(
                features, flip_pairs=None)

        if self.test_cfg.get('flip_test', True):
            img_flipped = img.flip(3)
            features_flipped = self.backbone(img_flipped)
            if self.with_neck:
                features_flipped = self.neck(features_flipped)
            if self.with_keypoint:
                output_flipped_heatmap = self.keypoint_head.inference_model(
                    features_flipped, img_metas[0]['flip_pairs'])
                output_heatmap = (output_heatmap +
                                  output_flipped_heatmap) * 0.5

        if self.with_keypoint:
            keypoint_result = self.keypoint_head.decode(
                img_metas, output_heatmap, img_size=[img_width, img_height])
            result.update(keypoint_result)

            if not return_heatmap:
                output_heatmap = None
            
            result['output_heatmap'] = output_heatmap

            if return_heatmap and hasattr(self, 'proto_head'):
                img_sources = torch.from_numpy(np.array([ele['dataset_idx'] for ele in img_metas])).to(img.device)
                unique_dset_idxs = torch.unique(img_sources)
                phms = []
                cphms = []
                for dset_idx in unique_dset_idxs:
                    dset_id = dset_idx.item()
                    dset_mask = img_sources==dset_idx
                    features_di = features[dset_mask]
                    res = self.proto_head.inference_model(features_di, None, dset_id)
                    flip_res = self.proto_head.inference_model(features_di, img_metas[0]['flip_pairs'], dset_id)

                    is_tuple = isinstance(res, tuple)

                    if is_tuple and len(res) == 2: # phm + cphm case
                        cphm = res[1]
                        """
                        cphm_flip = flip_res[1]
                        cphm = (cphm + cphm_flip) * 0.5
                        """
                        cphms.append(cphm)

                        phm = res[0]
                        phm_flip = flip_res[0]
                        #phm = (phm + phm_flip) * 0.5
                        phms.append(phm)
                    else:
                        phm = res
                        phm_flip = flip_res
                        #phm = (phm + phm_flip) * 0.5
                        phms.append(phm)
                if len(phms)==1:
                    result['proto_heatmap'] = phms[0]
                else:
                    result['proto_heatmap'] = phms

                if len(cphms) == 0:
                    pass
                elif len(cphms)==1:
                    result['cluster_heatmap'] = cphms[0]
                else:
                    result['cluster_heatmap'] = cphms
        
        return result

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            Tensor: Output heatmaps.
        """
        output = self.backbone(img)
        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            output = self.keypoint_head(output)
        return output

    @deprecated_api_warning({'pose_limb_color': 'pose_link_color'},
                            cls_name='TopDown')
    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_link_color=None,
                    text_color='white',
                    radius=4,
                    thickness=1,
                    font_scale=0.5,
                    bbox_thickness=1,
                    win_name='',
                    show=False,
                    show_keypoint_weight=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            skeleton (list[list]): The connection of keypoints.
                skeleton is 0-based indexing.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If None, do not draw links.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            show_keypoint_weight (bool): Whether to change the transparency
                using the predicted confidence scores of keypoints.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()

        bbox_result = []
        bbox_labels = []
        pose_result = []
        for res in result:
            if 'bbox' in res:
                bbox_result.append(res['bbox'])
                bbox_labels.append(res.get('label', None))
            pose_result.append(res['keypoints'])

        if bbox_result:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            imshow_bboxes(
                img,
                bboxes,
                labels=bbox_labels,
                colors=bbox_color,
                text_color=text_color,
                thickness=bbox_thickness,
                font_scale=font_scale,
                show=False)

        if pose_result:
            imshow_keypoints(img, pose_result, skeleton, kpt_score_thr,
                             pose_kpt_color, pose_link_color, radius,
                             thickness)

        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img

    @deprecated_api_warning({'pose_limb_color': 'pose_link_color'},
                            cls_name='TopDown')
    def show_result_pyplot(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_link_color=None,
                    text_color='white',
                    radius=4,
                    thickness=1,
                    font_scale=0.5,
                    bbox_thickness=1,
                    win_name='',
                    show=False,
                    show_keypoint_weight=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            skeleton (list[list]): The connection of keypoints.
                skeleton is 0-based indexing.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If None, do not draw links.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            show_keypoint_weight (bool): Whether to change the transparency
                using the predicted confidence scores of keypoints.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()
        img_rgb = img[:,:,::-1]
        imgplot = plt.imshow(img_rgb)
        plt.axis('off')

        bbox_result = []
        bbox_labels = []
        pose_result = []
        for res in result:
            if 'bbox' in res:
                bbox_result.append(res['bbox'])
                bbox_labels.append(res.get('label', None))
            pose_result.append(res['keypoints'])
        """
        if len(bbox_result)>0:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            imshow_bboxes_pyplot(
                imgplot,
                bboxes,
                labels=bbox_labels,
                colors=bbox_color,
                text_color=text_color,
                thickness=bbox_thickness,
                font_scale=font_scale,
                show=False)
        """
        
        if pose_result:
            imshow_keypoints_pyplot(imgplot, pose_result, skeleton, kpt_score_thr,
                             pose_kpt_color, pose_link_color, radius,
                             thickness)

        if show:
            imgplot.imshow(img, win_name, wait_time)

        if out_file is not None:
            #imgplot.imsave(out_file, format='pdf')
            imgplot.axes.figure.savefig(out_file, format='pdf', bbox_inches='tight')
            plt.clf()

        return img

