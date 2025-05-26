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

pw3d_scorehypo_dir = '../ScoreHypo/data/pw3d/annotations'
pw3d_train = read_joblib(osp.join(pw3d_scorehypo_dir, '3DPW_train.json'))
pw3d_test = read_joblib(osp.join(pw3d_scorehypo_dir, '3DPW_test.json'))
"""
pw3d_train['images'][0].keys():
['id', 'file_name', 'sequence', 'width', 'height', 'cam_param']

pw3d_train['annotations'][0].keys()
['id', 'image_id', 'fitted_3d_pose', 'smpl_param', 'bbox', 
    'h36m_joints', 'smpl_joint_img', 'smpl_joint_cam', 'joint_2d']

"""

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

        joint_img = ann['smpl_joint_img']
        joint_img = np.array(joint_img).reshape(24,3)
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
        #data['annotations'][ai]['segmentation'] = []

        path_rename_done = img_id_rename_done[img_id]
        if not path_rename_done:
            img_idx = img_id_to_idx[img_id]
            new_file_name = osp.join(img_ann['sequence'], img_ann['file_name'])
            data['images'][img_idx]['file_name'] = new_file_name
            img_id_rename_done[img_id] = True

        if 'fitted_3d_pose' in ann.keys():
            data['annotations'][ai].pop('fitted_3d_pose')
        data['annotations'][ai].pop('smpl_param')
        data['annotations'][ai].pop('h36m_joints')
        data['annotations'][ai].pop('smpl_joint_img')
        data['annotations'][ai].pop('smpl_joint_cam')
        if 'joint_2d' in ann.keys():
            data['annotations'][ai].pop('joint_2d')

    return data

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

new_pw3d_train = fit_annot(pw3d_train)
new_pw3d_train['categories'] = [category]
new_pw3d_train_path = 'data/pw3d/annotations/pw3d_train.json'
with open(new_pw3d_train_path, 'w') as f:
    json.dump(new_pw3d_train, f)

new_pw3d_test = fit_annot(pw3d_test)
new_pw3d_test['categories'] = [category]
new_pw3d_test_path = 'data/pw3d/annotations/pw3d_test.json'
with open(new_pw3d_test_path, 'w') as f:
    json.dump(new_pw3d_test, f)

print("Finished")
