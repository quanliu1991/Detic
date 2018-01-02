#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/2/16 2:05 下午
# @Author  : Liu Quan
# @File    : cala_ap.py
# @Software: PyCharm
"""
用于omstudio导出的json数据格式，转换为txt文件，用于AP计算。

txt格式
图片1.txt
类别 xmin ymin xmax ymax
...
"""


# 生成ground true
import argparse
from PIL import Image
import json
from os import listdir
import requests
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path


label_id_map={"制冰器门-关着":0,"制冰器门-开着":1,
              "水龙头-关着":2,"水龙头-开着":3,"垃圾桶-关着":4,
              "垃圾桶-开着":5,"芒果":6,"葡萄":7,"橙子":8,
              "柠檬":9,"椰子":10,"草莓":11,"西瓜":12,'案板':13, '塑料分装小盒':14,  '拖把':15, '店员':16, '金属茶桶':17, '水池':18,  '搅拌勺/棒':19,  '塑料分装壶/桶':20,'（贴在桶上的）保质期标签':21}

#{'葡萄', '草莓', '垃圾桶-开着', '芒果', '水龙头-关着', '橙子', '案板', '塑料分装小盒', '水龙头-开着', '拖把', '店员', '金属茶桶', '水池', '制冰器门-关着', '搅拌勺/棒', '西瓜', '塑料分装壶/桶', '制冰器门-开着', '（贴在桶上的）保质期标签', '垃圾桶-关着', '柠檬'}
# {'案板', '塑料分装小盒',  '拖把', '店员', '金属茶桶', '水池',  '搅拌勺/棒',  '塑料分装壶/桶',  '（贴在桶上的）保质期标签'}


def save_img_and_gen_txt(img_path,txt_path,img,name,pil,label_list,bb_list):
    w,h = pil.size
    img_save = img_path + '/' + name
    for i in range(len(bb_list)):
        this_bbox = bb_list[i]
        this_label = label_list[i]
        num_label = label_id_map[this_label]

        box =[this_bbox[0],this_bbox[1], this_bbox[2],this_bbox[3]]

        # Segments
        # segments = [j for i in x['segmentation'] for j in i]  # all segments concatenated
        # s = (np.array(segments).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()

        # Write
        if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
            img.save(img_save)
            line = tuple([num_label] + box)  # cls, box or segments
            new_name = name.split(".")[0] + ".txt"
            with open(txt_path + '/' + new_name, 'a') as file:
                # print(line)
                img.save(img_save)
                file.write(('%g ' * len(line)).rstrip() % line + '\n')
        else:
            print('bbox error')


def get_new_images_and_mask_bbox_without_crop_and_hit(input_list):
    counter =0
    for i in tqdm(range(len(input_list))):
        this_image = input_list[i]
        #pil_img = Image.open(urlopen(this_image['url'])).convert('RGB')
        pil_img = Image.open(requests.get(this_image['url'],stream=True).raw).convert('RGB')
        name=this_image['url'].split('/')[-1]
        mask_bb_list = [this_obj['bbox'] for this_obj in this_image['obj']]
        mask_label_list = [this_obj['object'] for this_obj in this_image['obj']]
        save_img_and_gen_txt(args.save_dir+'/images',
                             args.save_dir+'/gt',
                             pil_img,
                             name,
                             pil_img,
                             mask_label_list,
                             mask_bb_list)
        counter += 1




if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Start convert.')
    parser.add_argument('--soco_json_dir', type=str, help='soco json dir')
    parser.add_argument('--save_dir', type=str, help='conversion result save dir', default='../../datasets/mts_ch123')
    args = parser.parse_args()
    # datapath = args.soco_json_dir # json文件所在路径文件夹
    datapath=os.path.join(Path(os.path.dirname(__file__)).parent.parent,"datasets","mts_ch123")
    dir_list = [datapath + '/' + a for a in listdir(datapath)]
    print(dir_list)
    raw_data = []
    for dir in dir_list:
        if Path.is_file(Path(dir)):
            with open(dir) as f:
                this_raw_data = json.load(f)
            raw_data += this_raw_data
        get_new_images_and_mask_bbox_without_crop_and_hit(raw_data)
