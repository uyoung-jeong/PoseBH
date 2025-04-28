# preprocess raw apt-36k annotation to split train/val/test with 7:1:2 ratio
# apt-36k raw data should be stored in 'data/apt36k' directory
# preprocessed results are saved in 'prepro_out' folder
import json
import os
import os.path as osp
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
import copy

import sys
lib_path = osp.join(osp.dirname(__file__), '..', 'lib')
if lib_path not in sys.path:
        sys.path.insert(0, lib_path)

def write_json(data, json_path):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

data_base_dir = 'data/ap36k'
img_dir = osp.join(data_base_dir, 'images')

vis_dir = 'vis'
if not osp.exists(vis_dir):
    os.makedirs(vis_dir)

save_dir = 'prepro_out'
os.makedirs(save_dir, exist_ok=True)

raw_ann_path = osp.join(data_base_dir, 'annotations', 'apt36k_annotations.json')

with open(raw_ann_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

def initialize_dict(template_dict):
    categories = template_dict['categories']
    categories.sort(key = lambda x: x['id'])
    
    return_dict = {'license': copy.deepcopy(template_dict['license']),
                   'categories': copy.deepcopy(categories),
                   'info': copy.deepcopy(template_dict['info']),
                   'images': [],
                   'annotations': []}
    return return_dict

def write_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f)

train_dict = initialize_dict(raw_data)
val_dict = initialize_dict(raw_data)
test_dict = initialize_dict(raw_data)

# sample video ids uniformly over different categories
vids_per_cat = dict()
for img_ann in raw_data['images']:
    vid = img_ann['video_id']
    
    raw_file_path = img_ann['file_name']
    cat_name = raw_file_path.split('\\')[-3]

    if cat_name not in vids_per_cat:
        vids_per_cat[cat_name] = [vid]
    else:
        vids_per_cat[cat_name] += [vid]

train_vids = []
val_vids = []
test_vids = []

for k,v in vids_per_cat.items():
    num_vid = len(v)
    trainval_split = int((num_vid+1) * 0.7)
    valtest_split = int((num_vid+1) * 0.8)

    train_vids += v[:trainval_split]
    val_vids += v[trainval_split:valtest_split]
    test_vids += v[valtest_split:]

print(f"train videos: {len(train_vids)}, valid videos: {len(val_vids)}, test videos: {len(test_vids)}")

for img_ann in raw_data['images']:
    raw_file_path = img_ann['file_name']
    sub_path = raw_file_path.split('\\')[-3:]

    # handle filename encoding issue
    if sub_path[-2] == 'clip46√':
        sub_path[-2] = 'clip46б╠'
    elif sub_path[-2] == 'video11_clip6_1m44s-1m46s_frame╬╩╠Γ':
        sub_path[-2] = 'video11_clip6_1m44s-1m46s_frameиpиmиdжг'
    elif sub_path[-2] == 'video20_clip5_1m44s-1m46s_frame╬╩╠Γ':
        sub_path[-2] = 'video20_clip5_1m44s-1m46s_frameиpиmиdжг'
    elif sub_path[-2] == 'video2_clip3_0m55s-0m57s_frame╬╩╠Γ':
        sub_path[-2] = 'video2_clip3_0m55s-0m57s_frameиpиmиdжг'
    elif sub_path[-2] == 'v2c21太多':
        sub_path[-2] = 'v2c21╠л╢р'

    new_file_path = os.sep.join(sub_path)
    img_ann['file_name'] = new_file_path

    if img_ann['video_id'] in train_vids:
        train_dict['images'].append(img_ann)
    elif img_ann['video_id'] in val_vids:
        val_dict['images'].append(img_ann)
    else:
        test_dict['images'].append(img_ann)

for obj_ann in raw_data['annotations']:
    new_ann = dict()
    for k,v in obj_ann.items():
        if k == 'is_crowd':
            new_ann['iscrowd'] = v
        else:
            new_ann[k] = v
    if obj_ann['video_id'] in train_vids:
        train_dict['annotations'].append(new_ann)
    elif obj_ann['video_id'] in val_vids:
        val_dict['annotations'].append(new_ann)
    else:
        test_dict['annotations'].append(new_ann)

print(f"train images: {len(train_dict['images'])}, annotations: {len(train_dict['annotations'])}") # 24937, 36957
print(f"val images: {len(val_dict['images'])}, annotations: {len(val_dict['annotations'])}") # 3603, 5443
print(f"test images: {len(test_dict['images'])}, annotations: {len(test_dict['annotations'])}") # 6943, 10734

train_json_path = osp.join(save_dir, 'apt36k_train.json')
val_json_path = osp.join(save_dir, 'apt36k_val.json')
test_json_path = osp.join(save_dir, 'apt36k_test.json')

write_json(train_dict, train_json_path)
write_json(val_dict, val_json_path)
write_json(test_dict, test_json_path)

print(f"json files saved at {save_dir}")
