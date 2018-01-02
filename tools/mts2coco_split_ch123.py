#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import requests
import urllib
# import pandas as pd
from PIL import Image
import argparse
import copy
# from utils import *
import random
import os
import tqdm
from pathlib import Path
random.seed(1)

label_id_map={
    "制冰器门-关着":0,
    "制冰器门-开着":1,
    "水龙头-关着":2,
    "水龙头-开着":3,
    "垃圾桶-关着":4,
    "垃圾桶-开着":5,
    "芒果":6,
    "葡萄":7,
    "橙子":8,
    "柠檬":9,
    "椰子":10,
    "草莓":11,
    "西瓜":12,
    '案板':13,
    '塑料分装小盒':14,
    '拖把':15,
    '店员': 16,
    '金属茶桶':17,
    '水池':18,
    '搅拌勺/棒':19,
    '塑料分装壶/桶':20,
    "（贴在桶上的）保质期标签":21
}

label_zh_map_en={
    "制冰器门-关着":"ice_maker_door_closed",
    "制冰器门-开着":"ice_maker_door_open",
    "水龙头-关着":"tap_closed",
    "水龙头-开着":"tap_open",
    "垃圾桶-关着":"trash_can_closed",
    "垃圾桶-开着":"trash_can_open",
    "芒果":"mango",
    "葡萄":"grape",
    "橙子":"orange",
    "柠檬":"lemon",
    "椰子":"coconut",
    "草莓":"strawberry",
    "西瓜":"watermelon",
    '案板':"chopping_board",
    '塑料分装小盒':"capsule",
    '拖把':"mop",
    '店员': "clerk",
    '金属茶桶':"metal_tea_bucket",
    '水池':"pool",
    '搅拌勺/棒':"mixing_spoon",
    '塑料分装壶/桶':"separate_barrel",
    "（贴在桶上的）保质期标签":"shelf_life_label"
}


def normal_bbox(box):
    for i in range(4):
        if type(box[i]) == type({'$numberInt': '3000'}):
            box[i] = int(box[i]['$numberInt'])
    box[0] = max(box[0], 0)
    box[1] = max(box[1], 0)
    box[2] = max(box[2], 0)
    box[3] = max(box[3], 0)
    if box[0] > box[2]:
        box[0], box[2] = box[2], box[0]
    if box[1] > box[3]:
        box[1], box[3] = box[3], box[1]

    return box

def xyxy2xywh(box):
    x=box[0]
    y=box[1]
    w=box[2]-box[0]
    h=box[3]-box[1]
    return [x,y,w,h]


def annotations_extraction(datalist, category_map, raw_image_dir,ch_name,is_train):
    category_map_num = copy.deepcopy(category_map)
    for k in category_map_num.keys():
        category_map_num[k] = 0
    images = []
    annotations = []
    im_id = 0
    # try:
    #	  os.mkdir(save_dir)
    # except:
    #	  print('path exsits')
    for d in datalist:
        img_name=d['url'].split('/')[-1]
        file_img = os.path.join(raw_image_dir,img_name )
        if not os.path.isfile(file_img):
            print(f"{file_img} does not exist, check your local path!")
            continue
            import pdb
            pdb.set_trace()


        save_img_path=os.path.dirname(Path(raw_image_dir))
        mat = Image.open(file_img)
        if is_train:
            if ch_name=="ch1":
                mat.save(save_img_path+"/train_ch1"+"/"+img_name)
            if ch_name=="ch2":
                mat.save(save_img_path+"/train_ch2"+"/"+img_name)
            if ch_name=="ch3":
                mat.save(save_img_path+"/train_ch3"+"/"+img_name)
        else:
            if ch_name=="ch1":
                mat.save(save_img_path+"/val_ch1"+"/"+img_name)
            if ch_name=="ch2":
                mat.save(save_img_path+"/val_ch2"+"/"+img_name)
            if ch_name=="ch3":
                mat.save(save_img_path+"/val_ch3"+"/"+img_name)

        # f_io = requests.get(d['url']).content
        # # print(f_io)
        # mat = Image.open(io.BytesIO(f_io))
        # with open(os.path.join(save_dir, d['name'].split('/')[-1]), 'wb') as f:
        #	  f.write(f_io)
        # del f_io

        image = {
            "license": 0,
            "file_name": d['url'].split('/')[-1],
            "coco_url": d['url'],
            "height": mat.height,
            "width": mat.width,
            "id": im_id,
            "flickr_url": d['url'],
            "date_captured": None
        }
        # mat.close()
        for obj in d['obj']:
            category_map_num[label_zh_map_en[obj['object']]] += 1
            annotation = {
                "segmentation": [],
                "image_id": int(im_id),
                "bbox": xyxy2xywh(obj['bbox']),
                "category_id": int(category_map[label_zh_map_en[obj['object']]]),
                "id": int(im_id+1),
                "iscrowd": 0,
                "area": 5.5
            }
            annotations.append(annotation)
        images.append(image)
        im_id += 1
    return images, annotations, category_map_num



def convert_soco_to_coco(soco_jsonfile, raw_image_dir, save_dir, exclude_classes):
    '''
    soco_jsonfile : soco_jsonfile路径
    coco_jsondir: coco_json路径
    raw_image_dir: raw_image路径
    '''
    data = json.load(open(soco_jsonfile))
    category_map = {}
    cnt_null = 0
    exclude_classes = exclude_classes.split(',')

    for file in data:
        new_obj = []
        for obj in file['obj']:
            if len(obj['object']) == 0:
                cnt_null += 1
                print(f"{obj} does not have valid object annotation, skipping this object...")
                continue
            if len(exclude_classes) > 0 and obj['object'] in exclude_classes:
                print(f"{obj['object']} in {exclude_classes}, skipping this object..")
                continue
            category_map[label_zh_map_en[obj['object']]]=label_id_map[obj['object']]
            new_obj.append(obj)
        file['obj'] = new_obj
    print(category_map)
    print(f"{cnt_null} files have invalid(empty) object annotation.")

    os.makedirs(save_dir, exist_ok=True)

    fileList_ch1,fileList_ch2,fileList_ch3 = [],[],[]
    for data_ch in data:
        if "NVR_ch1_main" in data_ch["title"]:
            fileList_ch1.append(data_ch)
        elif "NVR_ch4_main_0" in data_ch["title"]:
            fileList_ch2.append(data_ch)
        elif "NVR_ch4_main_2022" in data_ch["title"]:
            fileList_ch3.append(data_ch)

    for ch_i,fileList in enumerate([fileList_ch1,fileList_ch2,fileList_ch3]):
        if ch_i==0:
            ch_name='ch1'
        if ch_i==1:
            ch_name='ch2'
        if ch_i==2:
            ch_name='ch3'

        random.shuffle(fileList)
        file_train = fileList[:int(len(fileList) * 0.9)]
        file_val = fileList[int(len(fileList) * 0.9):]


        images_train, annotations_train, category_map_num_train = annotations_extraction(file_train, category_map,
                                                                                         raw_image_dir,ch_name,True)
        images_val, annotations_val, category_map_num_val = annotations_extraction(file_val, category_map, raw_image_dir,ch_name,False)

        category_list = list(category_map.keys())
        categories = [{'supercategory': category_list[i], 'id': category_map[category_list[i]], 'name': category_list[i]}
                      for i in range(len(category_list))]
        print('category_map_num_train', category_map_num_train)
        print('category_map_num_val', category_map_num_val)

        # saved category map should be starting from 0
        # save_category_map = {k: v - 1 for k, v in category_map.items()}

        json.dump({'category_map': category_map, 'category_map_num_train': category_map_num_train,
                   'category_map_num_val': category_map_num_val},
                  open(os.path.join(save_dir, ch_name+"labels.json"), "w"), ensure_ascii=False, indent=2)
        D_train = {
            "info": {},
            "licenses": {},
            "images": images_train,
            "annotations": annotations_train,
            "categories": categories
        }

        D_val = {
            "info": {},
            "licenses": {},
            "images": images_val,
            "annotations": annotations_val,
            "categories": categories
        }

        coco_jsondir="/home/lq/projects/Detic/datasets/mts_ch123/annotations"
        os.makedirs(coco_jsondir, exist_ok=True)
        with open(coco_jsondir + '/instances_'+ ch_name +'_train.json', 'w', encoding='utf-8') as f:
              json.dump(D_train, f, ensure_ascii=False, indent=2)

        with open(coco_jsondir + '/instances_'+ ch_name +'_val.json', 'w', encoding='utf-8') as f:
              json.dump(D_val, f, ensure_ascii=False, indent=2)
    return True

if __name__ == '__main__':

    convert_soco_to_coco(soco_jsonfile='/home/lq/projects/Detic/datasets/mts_ch123/mts_0215.json',
						   raw_image_dir='/home/lq/projects/Detic/datasets/mts_ch123/images',
                           save_dir="/home/lq/projects/Detic/datasets/mts_new/",
                           exclude_classes="")
	  # convert_coco_json(json_dir='coco/', img_dir='/data0/public/data/2021-competition/tasks/室外消防通道占用/data-v1')


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Start convert.')
#     parser.add_argument('--soco_json_file', type=str, help='soco json file path')
#     parser.add_argument('--raw_image_dir', type=str, help='raw image data dir')
#     parser.add_argument('--exclude_classes', type=str, help='class names to exclude, sep by ","', default='')
#     parser.add_argument('--save_dir', type=str, help='save dir', default='save_dir')
#     args = parser.parse_args()
#     covert_soco_to_coco(args)