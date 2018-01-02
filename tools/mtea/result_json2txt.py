#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/2/17 9:01 下午
# @Author  : Liu Quan
# @File    : result_json2txt.py
# @Software: PyCharm
"""
用于预测的json文件，转换为txt文件，用于AP计算。
json格式。
{
  "result": [
    {
      "image_id": "NVR_ch1_main_20220101190000_20220101200000_00057.jpg",
      "instances": [
        {
          "box": [
            339.2731018066406,
            586.2799072265625,
            831.9044799804688,
            1015.3391723632812
          ],
          "class": 0,
          "scores": 0.8884528875350952
        }
    }
txt格式
{image_id}.txt
{class} {scores} xmin ymin xmax ymax
...
"""
import json
import os
from pathlib import Path
import argparse

from tqdm import tqdm


def json2txts(json_path):
    json_dir=os.path.dirname(json_path)
    predict_dir=os.path.join(Path(json_dir).parent,"predict")
    if not Path(predict_dir).exists():
        os.mkdir(predict_dir)
    with open(json_path,"r") as f:
        datas = json.load(f)["result"]

    for data in tqdm(datas):
        image_name=os.path.splitext(data["image_id"])[0]
        txt_name = os.path.join(predict_dir, image_name+".txt")
        for ins in data["instances"]:
            id=ins["class"]
            boxs=ins["box"]
            scores=ins["scores"]
            context=" ".join([str(id),str(scores),str(" ".join(list(map(str,boxs))))])
            try:
                save_txt(context,txt_name)
            except:
                print("save file fail")
    print("done!")

def save_txt(context,txt_name):
    with open(txt_name,"a") as f:
        f.write('%s '.rstrip() %context + '\n')



if __name__=="__main__":
    parse=argparse.ArgumentParser(description='json convert txt for cala AP.')
    parse.add_argument("--input_json",default="",type=str,help="predict json")
    parse.add_argument("--save_path",default="",type=str,help="txt files save path ")
    args=parse.parse_args()
    args.input_json="/home/lq/projects/Detic/datasets/mts_ch123/RPN_propose/result/prediction_result.json"
    json2txts(args.input_json)
