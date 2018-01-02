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
    '店员':"clerk",
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


def annotations_extraction(datalist, category_map, raw_image_dir):
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
        file_img = os.path.join(raw_image_dir, d['url'].split('/')[-1])
        if not os.path.isfile(file_img):
            print(f"{file_img} does not exist, check your local path!")
            continue
            import pdb
            pdb.set_trace()

        mat = Image.open(file_img)
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
        mat.close()
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

    category_set = set()
    category_map = {}
    new_fileList = []
    cnt_null = 0
    exclude_classes = exclude_classes.split(',')
    # for file in data['data'][0]['fileList']:
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
            category_set.add(label_zh_map_en[obj['object']])
            new_obj.append(obj)
        file['obj'] = new_obj
    category_list = list(category_set)
    print(category_list)
    print(f"{cnt_null} files have invalid(empty) object annotation.")
    for i in range(len(category_list)):
        category_map[category_list[i]] = i + 1
    os.makedirs(save_dir, exist_ok=True)

    # fileList = data['data'][0]['fileList']
    fileList = data



    # fileList = new_fileList
    random.shuffle(fileList)
    file_train = fileList[:int(len(fileList) * 0.8)]
    file_val = fileList[int(len(fileList) * 0.8):int(len(fileList) * 0.9)]
    file_test = fileList[int(len(fileList) * 0.9):]
    images_train, annotations_train, category_map_num_train = annotations_extraction(file_train, category_map,
                                                                                     raw_image_dir)
    images_val, annotations_val, category_map_num_val = annotations_extraction(file_val, category_map, raw_image_dir)
    images_test, annotations_test, category_map_num_test = annotations_extraction(file_test, category_map,
                                                                                  raw_image_dir)
    category_list = list(category_map.keys())
    categories = [{'supercategory': category_list[i], 'id': category_map[category_list[i]], 'name': category_list[i]}
                  for i in range(len(category_list))]
    print('category_map_num_train', category_map_num_train)
    print('category_map_num_val', category_map_num_val)
    print('category_map_num_test', category_map_num_test)
    # saved category map should be starting from 0
    save_category_map = {k: v - 1 for k, v in category_map.items()}

    json.dump({'category_map': save_category_map, 'category_map_num_train': category_map_num_train,
               'category_map_num_val': category_map_num_val, 'category_map_num_test': category_map_num_test},
              open(os.path.join(save_dir, "labels.json"), "w"), ensure_ascii=False, indent=2)
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

    D_test = {
        "info": {},
        "licenses": {},
        "images": images_test,
        "annotations": annotations_test,
        "categories": categories
    }
    coco_data = [D_train, D_val, D_test]


    coco_jsondir="/home/lq/projects/Detic/datasets/mts_ch123/annotations"
    os.makedirs(coco_jsondir, exist_ok=True)
    with open(coco_jsondir + '/instances_train.json', 'w', encoding='utf-8') as f:
          json.dump(D_train, f, ensure_ascii=False, indent=2)

    with open(coco_jsondir + '/instances_val.json', 'w', encoding='utf-8') as f:
          json.dump(D_val, f, ensure_ascii=False, indent=2)

    with open(coco_jsondir + '/instances_test.json', 'w', encoding='utf-8') as f:
          json.dump(D_test, f, ensure_ascii=False, indent=2)
    return coco_data

def convert_coco_json(raw_image_dir, coco_data, save_dir, use_segments=False):
    coco80 = coco91_to_coco80_class()

    # Import json
    dirs = ['train', 'val', 'test']
    for i, data in enumerate(coco_data):
        fn = os.path.join(save_dir, 'labels', dirs[i])
        os.makedirs(fn, exist_ok=True)
        fm = os.path.join(save_dir, 'images', dirs[i])
        os.makedirs(fm, exist_ok=True)

        # Create image dict
        images = {'%g' % x['id']: x for x in data['images']}

        for im in tqdm(data['images']):
            # urllib.request.urlretrieve(im['coco_url'], os.path.join(fm, im['file_name']))
            src_name = '/'.join(im['coco_url'].split('/')[-3:])
            if not os.path.isfile(src_path := os.path.join(raw_image_dir, src_name)):
                print(f"cannot find {src_path}, try to go down one path and find file...")
                src_name = '/'.join(im['coco_url'].split('/')[-2:])
                if not os.path.isfile(src_path := os.path.join(raw_image_dir, src_name)):
                    print(f"cannot find {src_path}, break...")
                    continue
                    # import pdb;
                    # pdb.set_trace()

            shutil.copy(src_path, os.path.join(fm, im['file_name']))
        # Write labels file
        for x in tqdm(data['annotations'], desc='Annotations %s'):  # % json_file):

            img = images['%g' % x['image_id']]
            h, w, f = img['height'], img['width'], img['file_name']

            # The COCO box format is [top left x, top left y, width, height]
            # x['bbox'] = [x['bbox'][0], x['bbox'][1], x['bbox'][2]-x['bbox'][0], x['bbox'][3]-x['bbox'][1]]
            x['bbox'][2:] = [x['bbox'][2] - x['bbox'][0], x['bbox'][3] - x['bbox'][1]]
            box = np.array(x['bbox'], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y
            # Segments
            segments = [j for i in x['segmentation'] for j in i]  # all segments concatenated
            s = (np.array(segments).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()

            # Write
            if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
                line = coco80[x['category_id'] - 1], *(s if use_segments else box)  # cls, box or segments
                new_name = f"{f.rsplit('.', 1)[0]}.txt"
                with open(os.path.join(fn, new_name), 'a') as file:
                    # print(line)
                    file.write(('%g ' * len(line)).rstrip() % line + '\n')


def covert_soco_to_coco(args):
    coco_data = convert_soco_to_coco(args.soco_json_file, args.raw_image_dir, args.save_dir, args.exclude_classes)
    convert_coco_json(args.raw_image_dir, coco_data, args.save_dir, use_segments=False)


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