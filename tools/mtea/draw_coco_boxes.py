#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/2/24 12:40 上午
# @Author  : Liu Quan
# @File    : draw_coco_boxes.py
# @Software: PyCharm
from pycocotools.coco import COCO
import cv2 as cv
import os
import numpy as np
import random

label_id_map = {"制冰器门-关着": 0, "制冰器门-开着": 1,
                "水龙头-关着": 2, "水龙头-开着": 3, "垃圾桶-关着": 4,
                "垃圾桶-开着": 5, "芒果": 6, "葡萄": 7, "橙子": 8,
                "柠檬": 9, "椰子": 10, "草莓": 11, "西瓜": 12, '案板': 13, '塑料分装小盒': 14,
                '拖把': 15, '店员': 16, '金属茶桶': 17, '水池': 18,
                '搅拌勺/棒': 19, '塑料分装壶/桶': 20, '（贴在桶上的）保质期标签': 21}

classes = ["door_closed","door_open","tap_closed","tap_open","trash_can_closed",
        "trash_can_open","mango","grape","orange","lemon",
        "coconut","strawberry","watermelon", "chopping_board", "capsule",
        "mop", "clerk", "metal_tea_bucket", "pool", "mixing_ spoon",
        "separate_barrel", "shelf_life_label"]

img_path = '/home/lq/projects/Detic/datasets/mts_ch123/train_ch1_ch2_ch3'  # 把图片直接放在同一文件夹下
annFile = '/home/lq/projects/Detic/datasets/mts_ch123/annotations/instances_ch1_ch2_ch3_train.json'  # 同样

coco = COCO(annFile)
cats = coco.loadCats(coco.getCatIds())

catIds = coco.getCatIds(catNms=[])
imgIds = coco.getImgIds(catIds=[])
img_list = os.listdir(img_path)
for i in range(len(img_list)):
    # img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
    img = coco.loadImgs(imgIds)[i]
    image_name = img['file_name']

    # 加了catIds就是只加载目标类别的anno，不加就是图像中所有的类别anno
    annIds = coco.getAnnIds(imgIds=[img['id']])
    anns = coco.loadAnns(annIds)

    coco.showAnns(anns)

    coordinates = []
    img_raw = cv.imread(os.path.join(img_path, image_name))

    font = cv.FONT_HERSHEY_SIMPLEX

    for label in range(0,22):

    for j in range(len(anns)):
        x1 = int(anns[j]['bbox'][0])
        y1 = int(anns[j]['bbox'][1] + anns[j]['bbox'][3])
        x2 = int(anns[j]['bbox'][0] + anns[j]['bbox'][2])
        y2 = int(anns[j]['bbox'][1])
        text = classes[anns[j]['category_id']]

        img_draw = cv.rectangle(img_raw,
                                (x1, y1),
                                (x2, y2),
                                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                                2)
        point = (x1, y1)
        id2label={}
        for k,v in label_id_map.items():
            id2label[v]=k


        img_draw = cv.putText(img_draw, str(text), point, font, 1, (255, 0, 0), 1)


    output_name = os.path.join(os.path.dirname(__file__), "box_output", text+".jpg",)
    cv.imwrite(output_name, img_draw)
    break
