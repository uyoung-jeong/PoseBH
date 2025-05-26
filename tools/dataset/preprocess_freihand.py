import json
import os
import os.path as osp
import numpy as np
from tqdm import tqdm

import cv2

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def dump_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

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

hand_category = [{'supercategory': 'hand', 'id': 1, 'name': 'hand', 
        'keypoints': ['wrist', 'thumb1', 'thumb2', 'thumb3', 'thumb4', 
                    'forefinger1', 'forefinger2', 'forefinger3', 'forefinger4', 'middle_finger1', 
                    'middle_finger2', 'middle_finger3', 'middle_finger4', 'ring_finger1', 'ring_finger2', 
                    'ring_finger3', 'ring_finger4', 'pinky_finger1', 'pinky_finger2', 'pinky_finger3', 'pinky_finger4'],
        'skeleton': [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], 
                    [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], 
                    [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], 
                    [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]}]

frei_base_dir = '/syn_mnt/uyoung/hand/FreiHAND'
frei_train_xyz = read_json(osp.join(frei_base_dir, 'training_xyz.json')) # [32560 x [21 x [3]]]
frei_train_mano = read_json(osp.join(frei_base_dir, 'training_mano.json')) # [32560 x [1 x [61]]]
frei_train_k = read_json(osp.join(frei_base_dir, 'training_K.json')) # [32560 x [3 x [3]]]
frei_train_scale = read_json(osp.join(frei_base_dir, 'training_scale.json')) # [32560]

frei_test_xyz = read_json(osp.join(frei_base_dir, 'evaluation_xyz.json')) # [3960 x [21 x [3]]]
frei_test_mano = read_json(osp.join(frei_base_dir, 'evaluation_mano.json')) # [3960 x [1 x [61]]]
frei_test_k = read_json(osp.join(frei_base_dir, 'evaluation_K.json')) # [3960 x [3 x [3]]]

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return img_coord

# https://github.com/lmb-freiburg/freihand/blob/master/utils/fh_utils.py
def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    #return uv[:, :2] / uv[:, -1:]
    return uv

def format_anno(xyz, mano, k, split='train'):
    new_dict = dict()
    new_dict['annotations'] = []
    new_dict['images'] = []
    num_ann = len(xyz)
    img_dir = osp.join(frei_base_dir, 'training/rgb') if split=='train' else osp.join(frei_base_dir, 'evaluation/rgb')
    img_reldir = 'training/rgb' if split=='train' else 'evaluation/rgb'
    radius = 5
    color = (0,0,255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.4

    imgid = 10000000
    annid = 10000000
    for ai in tqdm(range(num_ann)):
        img_path = osp.join(img_dir, f'{str(ai).zfill(8)}.jpg')
        img = cv2.imread(img_path)
        h,w = img.shape[:2]

        xyz_ai = xyz[ai]
        mano_ai = mano[ai]
        k_ai = k[ai]

        uvw_ai = projectPoints(xyz_ai, k_ai)
        uvw_ai[:,:2] = uvw_ai[:,:2] / uvw_ai[:,-1:]

        u_uf = uvw_ai[:,0]>=0
        v_uf = uvw_ai[:,1]>=0
        u_of = uvw_ai[:,0]<224
        v_of = uvw_ai[:,1]<224
        vis = u_uf * v_uf * u_of * v_of
        uvw_ai[:,2] = vis

        n_vis = int(vis.sum())
        min_x, min_y = np.amin(uvw_ai[:,0]), np.amin(uvw_ai[:,1])
        max_x, max_y = np.amax(uvw_ai[:,0]), np.amax(uvw_ai[:,1])
        
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        bbox_area = bbox_w * bbox_h

        # validate projection result
        if ai % 900 == 0 and split!='train':
            for kpt_i, kpt in enumerate(uvw_ai):
                kpt_vis = kpt[2]
                if kpt_vis == 0:
                    continue
                x_coord, y_coord = int(kpt[0]), int(kpt[1])
                img = cv2.circle(img, (x_coord, y_coord), radius, color, -1)
                img = cv2.putText(img, f'{kpt_i}', (x_coord+3, y_coord+3), font, fontScale, color, 1, cv2.LINE_AA)
            vis_path = osp.join('vis_results', f'{str(ai).zfill(8)}_vis.jpg')
            cv2.imwrite(vis_path, img)

        # img elem
        img_elem = {'file_name': osp.join(img_reldir, f'{str(ai).zfill(8)}.jpg'),
                    'height': h,
                    'width': w,
                    'id': imgid
        }
        # ann elem
        ann_elem = {'segmentation': [],
                    'num_keypoints': n_vis,
                    'area': bbox_area,
                    'iscrowd': 0,
                    'keypoints': uvw_ai.flatten().tolist(),
                    'image_id': imgid,
                    'bbox': [min_x, min_y, bbox_w, bbox_h],
                    'category_id': 1,
                    'id': annid
        }
        new_dict['annotations'].append(ann_elem)
        new_dict['images'].append(img_elem)
        imgid += 1
        annid += 1

    return new_dict

new_frei_train = format_anno(frei_train_xyz, frei_train_mano, frei_train_k, split='train')
new_frei_train['categories'] = hand_category
new_frei_train_path = 'data/freihand/annotations/freihand_2d_train.json'
dump_json(new_frei_train, new_frei_train_path)

new_frei_test = format_anno(frei_test_xyz, frei_test_mano, frei_test_k, split='test')
new_frei_test['categories'] = hand_category
new_frei_test_path = 'data/freihand/annotations/freihand_2d_test.json'
dump_json(new_frei_test, new_frei_test_path)

print("Finished")