#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/2/18 4:36 下午
# @Author  : Liu Quan
# @File    : cala_proposes_recall.py
# @Software: PyCharm
"""
calculate proposes recall,
inputs：propose_txt、ground_true_txt、iou_th
outputs：pre class recall
"""

# recall=numbers of peorposes cover instance/numbers of total instances


# 数据集中类别的总数量{calss_id:num}
# 被cover的类别总数量{class_id:num}

# 单张图片类别数量{calss_id:num}
# 单张图片被cover类别数量{calss_id:num}
import glob
import os
from pathlib import Path
# from Evaluator import *
from tqdm import tqdm


def get_all_boxs(directory,is_gt):
    all_bounding_boxes = {}
    all_classes = {}
    # Read ground truths
    os.chdir(directory)
    files = glob.glob("*.txt")
    files.sort()
    for f in tqdm(files):
        image_name = f.replace(".txt", "")
        with open(f, "r") as fh1:
            for line in fh1:
                line = line.replace("\n", "")
                if line.replace(' ', '') == '':
                    continue
                splitLine = line.split(" ")
                if is_gt:
                    class_id = (splitLine[0])
                    xmin = float(splitLine[1])
                    ymin = float(splitLine[2])
                    xmax = float(splitLine[3])
                    ymax = float(splitLine[4])
                    bb=BoundingBox(image_name,class_id,xmin,ymin,xmax,ymax)
                else:
                    class_id = (splitLine[0])
                    xmin = float(splitLine[2])
                    ymin = float(splitLine[3])
                    xmax = float(splitLine[4])
                    ymax = float(splitLine[5])
                    bb = BoundingBox(image_name, class_id, xmin, ymin, xmax, ymax)

                if image_name not in all_bounding_boxes:
                    all_bounding_boxes[image_name]=[bb]
                else:
                    all_bounding_boxes[image_name].append(bb)

                if class_id not in all_classes:
                    all_classes[class_id]=1
                else:
                    all_classes[class_id]+=1
    return (all_bounding_boxes,all_classes) if is_gt else all_bounding_boxes

def calculate_conver_nums_classes(gt_boxes,pp_boxes,iou_th):
    conver_nums_classes={}
    for img_name,gt_bb in tqdm(gt_boxes.items()):
        pp_bb=pp_boxes[img_name]
        for gt_b in gt_bb:
            for pp_b in pp_bb:
                if is_covered(gt_b.box,pp_b.box,iou_th):
                    if gt_b.class_id not in conver_nums_classes:
                        conver_nums_classes[gt_b.class_id]=1
                    else:
                        conver_nums_classes[gt_b.class_id] += 1
                    break
                else:
                    continue
    return conver_nums_classes


def calculate_recall(conver_nums_classes,all_classes):
    recalls={}
    for class_id in all_classes:
        try:
            recalls[class_id]=conver_nums_classes[class_id]/all_classes[class_id]
        except:
            print(class_id,"is not ")
    print(recalls)
    print(conver_nums_classes)
    print(all_classes)
    return recalls

class BoundingBox:
    def __init__(self,image_name,class_id,xmin,ymin,xmax,ymax):
        self.image_name=image_name
        self.class_id=class_id
        self.box=[xmin,ymin,xmax,ymax]

def is_covered(rec1,rec2,iou_th):
    """

    :param rec1:
    :param rec2:
    :param iou_th:
    :return: bool
    """
    if compute_iou(rec1, rec2) > iou_th:
        return True
    else:
        return False


def compute_iou(rec1,rec2):
    cross_left=max(rec1[0],rec2[0])
    cross_up=max(rec1[1],rec2[1])
    cross_right=min(rec1[2],rec2[2])
    cross_down=min(rec1[3],rec2[3])
    rec_cross=[cross_left,cross_up,cross_right,cross_down]

    if cross_left>=cross_right or cross_up>=cross_down:
        return 0
    else:
        def compute_rec_area(rec):
            area=(rec[2]-rec[0])*(rec[3]-rec[1])
            return area
        area_1=compute_rec_area(rec1)
        area_2=compute_rec_area(rec2)
        area_cross=compute_rec_area(rec_cross)
        return area_cross/(area_1+area_2-area_cross)






if __name__=="__main__":
    gt_bounding_boxes,all_classes=get_all_boxs("/home/lq/projects/Detic/datasets/mts_ch123/gt",True)
    pp_bounding_boxes = get_all_boxs("/home/lq/projects/Detic/datasets/mts_ch123/RPN_propose/result/propose_box", False)
    conver_nums_classes=calculate_conver_nums_classes(gt_bounding_boxes,pp_bounding_boxes,0.50)
    calculate_recall(conver_nums_classes , all_classes)
    print("done!")

