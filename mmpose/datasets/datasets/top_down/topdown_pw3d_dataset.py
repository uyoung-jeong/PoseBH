# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv import Config
from xtcocotools.cocoeval import COCOeval

from ...builder import DATASETS
from .topdown_coco_dataset import TopDownCocoDataset


@DATASETS.register_module()
class TopDownPW3DDataset(TopDownCocoDataset):
    """PW3DDataset dataset for 2D top-down pose estimation.

    "Recovering Accurate 3D Human Pose in The Wild Using IMUs and a Moving Camera", arXiv'2017.
    More details can be found in the `paper
    <https://virtualhumans.mpi-inf.mpg.de/papers/vonmarcardECCV18/vonmarcardECCV18.pdf>`__

    The dataset loads 2D-projected SMPL keypoints.

    3DPW keypoint indexes::
        'pelvis': 0,
        'left_hip': 1,
        'right_hip': 2,
        'spine1': 3,
        'left_knee': 4,
        'right_knee': 5,
        'spine2': 6,
        'left_ankle': 7,
        'right_ankle': 8,
        'spine3': 9,
        'left_foot': 10,
        'right_foot': 11,
        'neck': 12,
        'left_collar': 13,
        'right_collar': 14,
        'jaw': 15,
        'left_shoulder': 16,
        'right_shoulder': 17,
        'left_elbow': 18,
        'right_elbow': 19,
        'left_wrist': 20,
        'right_wrist': 21,
        'left_thumb': 22,
        'right_thumb': 23

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/pw3d.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super(TopDownCocoDataset, self).__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.use_gt_bbox = data_cfg['use_gt_bbox']
        self.bbox_file = data_cfg['bbox_file']
        self.det_bbox_thr = data_cfg.get('det_bbox_thr', 0.0)
        self.use_nms = data_cfg.get('use_nms', True)
        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']
        
        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        assert self.use_gt_bbox
        gt_db = self._load_coco_keypoint_annotations()
        return gt_db

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""
        coco_det = self.coco.loadRes(res_file)
        coco_eval = COCOeval(
            self.coco, coco_det, 'keypoints', self.sigmas, use_area=False)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str
