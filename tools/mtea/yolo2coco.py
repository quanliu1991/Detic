import imp
import json
import cv2
import os
import argparse

"""fewshot"""

#第一个coco json类型

bnd_id_start = 0

times = 0

json_dict = {
    "images"     : [],
    "type"       : "instances",
    "annotations": [],
    "categories" : []
}


# 这里是你的txt文件的读取
# with open('train.txt','r') as f:
#     data = f.readlines()

parser=argparse.ArgumentParser(description="yolo2coco")
parser.add_argument("--images_path",default="",help="images path")
parser.add_argument("--labels_path",default="",help="label path")
parser.add_argument("--save_name",default="",help="save ann file name")
args=parser.parse_args()
# raw_images_path = args.images_path # '/data0/public/data/mts_0215_1/images'
raw_images_path = '/home/lq/datasets/mts_0216_three_scene/ch1/val/images'
# raw_labels_path = args.labels_path # '/data0/public/data/mts_0215_1/labels'
raw_labels_path = '/home/lq/datasets/mts_0216_three_scene/ch1/val/labels'
save_file_name = args.save_name
save_file_name = '/home/lq/datasets/fewshot_demo'
data = os.listdir(raw_images_path)


bnd_id = bnd_id_start

#类别的名字(cid,cate)对应
# classes = ["door_closed","door_open","tap_closed","tap_open","trash_can_closed",
#         "trash_can_open","mango","grape","orange","lemon",
#         "coconut","strawberry","watermelon", "chopping_board", "capsule",
#         "mop", "clerk", "metal_tea_bucket", "pool", "mixing_ spoon",
#         "separate_barrel", "shelf_life_label"]

classes = ["clerk", "metal_tea_bucket", "pool", "mixing_spoon","separate_barrel"]

# classes_count = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0}
classes_count = {0:0, 1:0,2:0,3:0,4:0}
classes_images_count={0:0, 1:0,2:0,3:0,4:0}

for d in data:
    filename = d     #这里可能修改，txt文件每一行第一个属性是图片路径，通过split()函数把图像名分离出来就行
    img = cv2.imread(os.path.join(raw_images_path, filename))
    txtFile = filename.replace('jpeg', 'txt').replace('jpg', 'txt')
    # import pdb;pdb.set_trace()
    try:
        height,width = img.shape[0],img.shape[1]
        image_id = filename.split(".")[0]
    except:
        times += 1
        print('file is error')

    images_path=os.path.join(save_file_name,"val_5")
    if not os.path.exists(images_path):
        os.mkdir(images_path)
    image_path=os.path.join(save_file_name,"val_5",filename)
# type 已经填充

#定义image 填充到images里面

    is_images_count=[True]*5
    with open(os.path.join(raw_labels_path, txtFile), 'r') as fr:
        labelList = fr.readlines()
        for c in labelList:
            
            label, xmin, ymin, w, h = c.strip().split(" ")
            label = int(label)
            if label in [16,17,18,19,20]:
                if is_images_count[label-16] and classes_images_count[label-16]<401:
                    classes_images_count[label-16]+=1
                    is_images_count[label - 16]=False
                if  classes_images_count[label-16]<401:
                    classes_count[label - 16] += 1
                    # import pdb;pdb.set_trace()
                    xmin = float(xmin)
                    ymin = float(ymin)
                    w = float(w)
                    h = float(h)
                    x1 = width * xmin - 0.5 * width * w
                    y1 = height * ymin - 0.5 * height * h
                    x2 = width * xmin + 0.5 * width * w
                    y2 = height * ymin + 0.5 * height * h
                    o_width = abs(x2 - x1)
                    o_height = abs(y2 - y1)

                    area = o_width * o_height

                    # #定义annotation
                    annotation = {
                        'area'          : area,  #
                        'iscrowd'       : 0,
                        'image_id'      : image_id,  #图片的id
                        'bbox'          :[x1, y1, o_width,o_height],
                        'category_id'   : label-16, #类别的id 通过这个id去查找category里面的name
                        'id'            : bnd_id,  #唯一id ,可以理解为一个框一个Id
                        'ignore'        : 0,
                        'segmentation'  : [[x1, y1, x1 + o_width, y1, x1 + o_width, y1 + o_height, x1, y1 + o_height]]
                    }


                    json_dict['annotations'].append(annotation)

                    bnd_id += 1


                    if not os.path.exists(image_path):
                        cv2.imwrite(image_path,img)

                    image = {
                        'file_name': filename,  # 文件名
                        'height': height,  # 图片的高
                        'width': width,  # 图片的宽
                        'id': image_id  # 图片的id，和图片名对应的
                    }

                    flag=True
                    for json_image in json_dict['images']:
                        if filename ==json_image['file_name']:
                            flag=False
                    if flag:
                        json_dict['images'].append(image)

#定义categories



for i in range(len(classes)):

    cate = classes[i]
    cid = i
    category = {
        'supercategory' : 'none',
        'id'            : cid,  #类别的id ,一个索引，主键作用，和别的字段之间的桥梁
        'name'          : cate  #类别的名字
    }

    json_dict['categories'].append(category)



json_fp = open(save_file_name+"/instance_val_5.json",'w')
json_str = json.dumps(json_dict, indent=4)
json_fp.write(json_str)
json_fp.close()

print(classes_count)
print(classes_images_count)
