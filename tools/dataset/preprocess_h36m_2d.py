import json
import joblib
import os
import os.path as osp
import numpy as np
from tqdm import tqdm

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def dump_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

def read_joblib(path):
    with open(path, 'rb') as f:
        data = json.load(f)
    return data

# load coco annotation as a reference
coco_ann_path = 'data/coco/annotations/person_keypoints_val2017.json'
coco = read_json(coco_ann_path)
"""
coco['images'][0].keys():
['license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id']

coco['annotations'][0].keys():
['segmentation', 'num_keypoints', 'area', 'iscrowd', 'keypoints', 
    'image_id', 'bbox', 'category_id', 'id']

"""

h36m_scorehypo_dir = '../ScoreHypo/data/h36m/annotations'
h36m_train = read_joblib(osp.join(h36m_scorehypo_dir, 'Sample_5_train_Human36M_smpl_leaf_twist_protocol_2.json'))
h36m_test = read_joblib(osp.join(h36m_scorehypo_dir, 'Sample_20_test_Human36M_smpl_protocol_2.json'))
"""
h36m_train['images'][0].keys():
['id', 'file_name', 'width', 'height', 'subject', 'action_name', 'action_idx', 'subaction_idx', 'cam_idx', 'frame_idx', 'cam_param']
 
h36m_train['annotations'][0].keys():
['thetas', 'betas', 'bbox', 'area', 'iscrowd', 'category_id', 'image_id', 'id', 'root_coord', 'h36m_joints', 'smpl_joints', 'angle_twist']

h36m_train['categories'][0]:
{'supercategory': 'person', 'id': 1, 'name': 'person', 
'H36M_TO_J17': [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9], 
'H36M_TO_J14': [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10], 
'J24_TO_J17': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17], 
'J24_TO_J14': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18], 
'SMPL_JOINTS_FLIP_PERM': [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22], 
'J24_FLIP_PERM': [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18, 19, 21, 20, 23, 22]}

h36m_test['annotations'][0].keys():
['thetas', 'betas', 'bbox', 'area', 'iscrowd', 'category_id', 'image_id', 'id', 'root_coord', 'smpl_joints', 'h36m_joints']

"""

category = coco['categories'][0]
category['keypoints'] = [
        'pelvis', 'left_hip', 'right_hip', 
        'spine1', 'left_knee', 'right_knee', 
        'spine2', 'left_ankle', 'right_ankle', 
        'spine3', 'left_foot', 'right_foot', 
        'neck', 'left_collar', 'right_collar', 
        'jaw', 
        'left_shoulder', 'right_shoulder', 
        'left_elbow', 'right_elbow', 
        'left_wrist', 'right_wrist', 
        'left_thumb', 'right_thumb'
    ]
category['skeleton'] = [[7, 8], [4, 1], [8, 5], [5, 2], [1, 0],
                        [2, 0], [0, 3], [3, 6], [6, 9], [9, 12],
                        [12, 15], [12, 16], [12, 17], [16, 18], [17, 19],
                        [18, 20], [19, 21], [20, 22], [21, 23], [7, 10], [8, 11]]

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return img_coord

def fit_annot(data):
    img_id_to_img_ann = {e['id']:e for e in data['images']}
    img_id_to_idx = {e['id']:idx for idx,e in enumerate(data['images'])}
    img_id_rename_done = {e['id']:False for e in data['images']}
    num_ann = len(data['annotations'])
    for ai in tqdm(range(num_ann)):
        ann = data['annotations'][ai]
        img_id = ann['image_id']
        img_ann = img_id_to_img_ann[img_id]
        img_h = img_ann['height']
        img_w = img_ann['width']

        f, c = np.array(img_ann['cam_param']['f'], dtype=np.float32), np.array(
                img_ann['cam_param']['c'], dtype=np.float32)
        joint_cam = np.array(ann['smpl_joints']).reshape(-1,3) # train: [29,3], test: [24,3]
        joint_cam = joint_cam[:24] # use 24 joints.
        joint_img = cam2pixel(joint_cam, f, c)

        vis_x = (joint_img[:,0] >= 0.) * (joint_img[:,0] < img_w)
        vis_y = (joint_img[:,1] >= 0.) * (joint_img[:,1] < img_h)
        vis_d = joint_img[:,2] >= 0.
        vis = vis_x * vis_y * vis_d

        num_vis = int(vis.sum())
        joint_img[:,2] = vis

        data['annotations'][ai]['keypoints'] = joint_img.flatten().tolist()
        data['annotations'][ai]['num_keypoints'] = num_vis
        data['annotations'][ai]['iscrowd'] = 0
        data['annotations'][ai]['category_id'] = 1

    return data

new_h36m_train = fit_annot(h36m_train)
new_h36m_train['categories'] = [category]
new_h36m_train_path = 'data/h36m/annotations/h36m_2d_train.json'
dump_json(new_h36m_train, new_h36m_train_path)

new_h36m_test = fit_annot(h36m_test)
new_h36m_test['categories'] = [category]
new_h36m_test_path = 'data/h36m/annotations/h36m_2d_test.json'
dump_json(new_h36m_test, new_h36m_test_path)

print("Finished")
