#!/usr/bin/env python
# -*- coding: utf-8 -*-
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# def get_mts_dicts()
register_coco_instances("mts_train_ch1", {},
                        "/home/lq/projects/Detic/datasets/mts_ch123/annotations/instances_ch1_train.json",
                        "/home/lq/projects/Detic/datasets/mts_ch123/train_ch1")
register_coco_instances("mts_train_ch2", {},
                        "/home/lq/projects/Detic/datasets/mts_ch123/annotations/instances_ch2_train.json",
                        "/home/lq/projects/Detic/datasets/mts_ch123/train_ch2")
register_coco_instances("mts_train_ch3", {},
                        "/home/lq/projects/Detic/datasets/mts_ch123/annotations/instances_ch3_train.json",
                        "/home/lq/projects/Detic/datasets/mts_ch123/train_ch3")
register_coco_instances("mts_val_ch1", {},
                        "/home/lq/projects/Detic/datasets/mts_ch123/annotations/instances_ch1_val.json",
                        "/home/lq/projects/Detic/datasets/mts_ch123/val_ch1")
register_coco_instances("mts_val_ch2", {},
                        "/home/lq/projects/Detic/datasets/mts_ch123/annotations/instances_ch2_val.json",
                        "/home/lq/projects/Detic/datasets/mts_ch123/val_ch2")
register_coco_instances("mts_val_ch3", {},
                        "/home/lq/projects/Detic/datasets/mts_ch123/annotations/instances_ch3_val.json",
                        "/home/lq/projects/Detic/datasets/mts_ch123/val_ch3")
register_coco_instances("mts_train_ch1_ch2", {},
                        "/home/lq/projects/Detic/datasets/mts_ch123/annotations/instances_ch1_ch2_train.json",
                        "/home/lq/projects/Detic/datasets/mts_ch123/train_ch1_ch2")
register_coco_instances("mts_train_ch1_ch2_ch3", {},
                        "/home/lq/projects/Detic/datasets/mts_ch123/annotations/instances_ch1_ch2_ch3_train.json",
                        "/home/lq/projects/Detic/datasets/mts_ch123/train_ch1_ch2_ch3")
register_coco_instances("mts_train_fewshot", {}, "/home/lq/datasets/fewshot_demo/instance_train.json",
                        "/home/lq/datasets/fewshot_demo/train")
register_coco_instances("mts_val_fewshot", {}, "/home/lq/datasets/fewshot_demo/instance_val.json",
                        "/home/lq/datasets/fewshot_demo/val")
register_coco_instances("mts_train_fewshot_50", {}, "/home/lq/datasets/fewshot_demo/instance_train_50.json",
                        "/home/lq/datasets/fewshot_demo/train_50")
register_coco_instances("mts_train_fewshot_100", {}, "/home/lq/datasets/fewshot_demo/instance_train_100.json",
                        "/home/lq/datasets/fewshot_demo/train_100")
register_coco_instances("mts_train_fewshot_150", {}, "/home/lq/datasets/fewshot_demo/instance_train_150.json",
                        "/home/lq/datasets/fewshot_demo/train_150")
register_coco_instances("mts_train_fewshot_200", {}, "/home/lq/datasets/fewshot_demo/instance_train_200.json",
                        "/home/lq/datasets/fewshot_demo/train_200")
register_coco_instances("mts_train_fewshot_250", {}, "/home/lq/datasets/fewshot_demo/instance_train_250.json",
                        "/home/lq/datasets/fewshot_demo/train_250")
register_coco_instances("mts_train_fewshot_300", {}, "/home/lq/datasets/fewshot_demo/instance_train_300.json",
                        "/home/lq/datasets/fewshot_demo/train_300")
register_coco_instances("mts_train_fewshot_350", {}, "/home/lq/datasets/fewshot_demo/instance_train_350.json",
                        "/home/lq/datasets/fewshot_demo/train_350")
register_coco_instances("mts_train_fewshot_400", {}, "/home/lq/datasets/fewshot_demo/instance_train_400.json",
                        "/home/lq/datasets/fewshot_demo/train_400")
register_coco_instances("mts_val_fewshot_5", {}, "/home/lq/datasets/fewshot_demo/instance_val_5.json",
                        "/home/lq/datasets/fewshot_demo/val_5")

register_coco_instances("mts_val_fewshot_5_1", {}, "/home/lq/datasets/fewshot_demo/instance_val_5_1.json",
                        "/home/lq/datasets/fewshot_demo/val_5")

MetadataCatalog.get("mts_val_fewshot_5_1").set(thing_classes=[
        "person","bucket","basin","stirring rod","cup"
    ])

register_coco_instances("mts_val_fewshot_5_2", {}, "/home/lq/datasets/fewshot_demo/instance_val_5_2.json",
                        "/home/lq/datasets/fewshot_demo/val_5")

MetadataCatalog.get("mts_val_fewshot_5_2").set(thing_classes=[
        "man","barrel","vegatable basin","muddler","glass"
    ])

register_coco_instances("mts_val_fewshot_5_3", {}, "/home/lq/datasets/fewshot_demo/instance_val_5_3.json",
                        "/home/lq/datasets/fewshot_demo/val_5")

MetadataCatalog.get("mts_val_fewshot_5_3").set(thing_classes=[
        "people","pail","lavabo","stirrer","tumbler"
    ])

register_coco_instances("mts_val_fewshot_5_max", {}, "/home/lq/datasets/fewshot_demo/instance_val_5_max.json",
                        "/home/lq/datasets/fewshot_demo/val_5")

MetadataCatalog.get("mts_val_fewshot_5_max").set(thing_classes=[
        "clerk","pail","basin","mixing_spoon","tumbler"
    ])


for dataset in ["mts_train_fewshot_50", "mts_train_fewshot_150", "mts_train_fewshot_250", "mts_train_fewshot_350",
                "mts_train_fewshot_100", "mts_train_fewshot_200", "mts_train_fewshot_300", "mts_train_fewshot_400",
                "mts_val_fewshot_5"]:
    MetadataCatalog.get(dataset).set(thing_classes=[
        "clerk",
        "metal_tea_bucket",
        "pool",
        "mixing_spoon",
        "separate_barrel"
    ])



for dataset in ["mts_train_fewshot", "mts_val_fewshot"]:
    MetadataCatalog.get(dataset).set(thing_classes=[
        "trash_can",
        "mango"
    ])

# MetadataCatalog.get("mts_train").set(thing_classes=['草莓', '水龙头-开着', '葡萄', '芒果', '西瓜', '水龙头-关着', '垃圾桶-关着', '制冰器门-开着', '橙子', '制冰器门-关着', '椰子', '柠檬', '垃圾桶-开着'])
for dataset in ["mts_train_ch1", "mts_train_ch1", "mts_train_ch3", "mts_train_ch1_ch2", "mts_train_ch1_ch2_ch3",
                "mts_val_ch1", "mts_val_ch2", "mts_val_ch3"]:
    MetadataCatalog.get(dataset).set(thing_classes=[
        "door_closed",
        "door_open",
        "tap_closed",
        "tap_open",
        "trash_can_closed",
        "trash_can_open",
        "mango",
        "grape",
        "orange",
        "lemon",
        "coconut",
        "strawberry",
        "watermelon",
        "chopping_board",
        "capsule",
        "mop",
        "clerk",
        "metal_tea_bucket",
        "pool",
        "mixing_ spoon",
        "separate_barrel",
        "shelf_life_label"
    ])
# print(MetadataCatalog.get("mts_train"))
# print("a")

# ['ice_maker_door_closed', 'ice_maker_door_open', 'tap_closed', 'tap_open', 'trash_can_closed', 'trash_can_open', 'mango', 'grape', 'orange', 'lemon', 'strawberry', 'watermelon', 'chopping_board', 'capsule', 'mop', 'clerk', 'metal_tea_bucket', 'pool', 'mixing_spoon', 'separate_barrel', 'shelf_life_label'] !=
# ['ice_maker_door-closed', 'ice_maker_door-open', 'tap_closed', 'tap_open', 'trash_can_closed', 'trash_can_open', 'mango', 'grape', 'orange', 'lemon', 'strawberry', 'watermelon', 'chopping_board', 'capsule', 'mop', 'clerk', 'metal_tea_bucket', 'pool', 'mixing_spoon', 'separate_barrel', 'shelf_life_label']

# ['door_closed', 'door_open', 'tap_closed', 'tap_open', 'trash_can_closed', 'trash_can_open', 'mango', 'grape', 'orange', 'lemon', 'coconut', 'strawberry', 'watermelon', 'chopping_board', 'capsule', 'mop', 'clerk', 'metal_tea_bucket', 'pool', 'mixing_spoon', 'separate_barrel', 'shelf_life_label']
# ['door_closed', 'door_open', 'tap_closed', 'tap_open', 'trash_can_closed', 'trash_can_open', 'mango', 'grape', 'orange', 'lemon', 'coconut', 'strawberry', 'watermelon', 'chopping_board', 'capsule', 'mop', 'clerk', 'metal_tea_bucket', 'pool', 'mixing_spoon', 'separate_barrel', 'shelf_life_label']
