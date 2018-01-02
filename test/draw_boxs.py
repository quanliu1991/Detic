#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/2/13 2:03 下午
# @Author  : Liu Quan
# @File    : draw_boxs.py
# @Software: PyCharm
import json

from PIL import Image,ImageDraw


def draw_bbox(image, xmin, ymin, xmax, ymax, color='red', text='', width=4):
    draw = ImageDraw.Draw(image)
    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=width)
    draw.text((xmin, ymin), text)
    return image

def load_result_json(json_file):
    with open(json_file, 'r') as load_f:
        d=json.load(load_f)



def load_image(img_path):
    img = Image.open(img_path)

if __name__=="__main__":
    json_file="/home/lq/projects/Detic/output/Detic/Detic_LbaseI_CLIP_R5021k_640b64_4x_ft4x_max-size/inference_mts_train/coco_instances_results.json"
    load_result_json(json_file)