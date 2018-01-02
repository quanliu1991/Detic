#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/2/20 12:20 上午
# @Author  : Liu Quan
# @File    : draw_box.py
# @Software: PyCharm
from PIL import ImageDraw,Image


def draw_bbox(image, xmin, ymin, xmax, ymax, color='red', text='', width=4):
    draw = ImageDraw.Draw(image)
    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=width)
    draw.text((xmin, ymin), text)
    return image


data="/home/lq/projects/Detic/datasets/mts_ch123/images/NVR_ch1_main_20220103200000_20220103210000_00703.jpg"
outdata="/home/lq/projects/Detic/out5.jpeg"
image = Image.open(data)
xmin, ymin, xmax, ymax=276.99177096012403,818.3723272365502,589.7244155925223,993.4718604117245

draw_bbox(image, xmin, ymin, xmax, ymax)
image.save(outdata)

# x方向是w
# y方向是h
