#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/2/16 4:26 下午
# @Author  : Liu Quan
# @File    : fruit_text.py
# @Software: PyCharm

fruit_text="Apple 苹果 Banana 香蕉 Cherry 樱桃 Cherry_tomato 圣女果 Chestnut 栗子 Coconut 椰子 Cucumber 黄瓜 Cumquat 金桔 Date 枣子 Durian 榴莲 Grape 葡萄 Grapefruit 葡萄柚子 Guava 番石榴 Haw 山楂 Honey_dew_melon 哈蜜瓜 Juicy_peach 水蜜桃 Kiwifruit 猕猴桃 Lemon 柠檬 Lichee 荔枝 Longan 龙眼 Loquat 枇杷 Mandarin 中国柑桔 Mango 芒果 Mangosteen 山竹果 Mini_watermelon 小西瓜 Nectarine 油桃 Nucleus 核仁 Orange 橙子 Papaya 木瓜 Peach 桃子 Pear 梨 Persimmon 柿子 Pineapple 菠萝 Pitaya 火龙果 Pomegranate 石榴 Pomelo 柚子 Strawberry 草莓 Sugarcane 甘蔗 Tangerine 蜜柑桔 Warden 冬梨 Watermelon 西瓜"

def process_text(text):
    fruit_en=[]
    fruit_ch=[]
    fruit_en_ch_map={}
    fruit_ch_en_map={}
    fruit_text=text.split(" ")
    for fruit in fruit_text:
        en,ch=map(str,fruit.split("\xa0"))
        fruit_en.append(en.lower())
        fruit_ch.append(ch)
        fruit_en_ch_map[en]=ch
        fruit_ch_en_map[ch]=en
    fruit_en_txt=",".join(fruit_en)
    print(fruit_en_txt)
    print(fruit_en_ch_map)

process_text(fruit_text)


