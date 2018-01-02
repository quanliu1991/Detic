from ast import arg
import imp
import json
from os.path import join as opj
import os
from turtle import down
from PIL import Image
import argparse
import copy
import random
from joblib import Parallel, delayed
import uuid
from detectron2.data import detection_utils as utils
from tqdm import tqdm
import string
import urllib
import urllib.request
from urllib.parse import quote
from tqdm.contrib import tzip
"""
python studio2coco.py --studio_json_file <studio json path> --save_dir <save images and annotations path > --exclude_classes <(optional)> --dataset_name <数据名>
example: python studio2coco.py --studio_json_file /home/ljj/video-tools/temp_data/001_vg_train_1635470915892.json --save_dir /home/ljj/video-tools --dataset_name 001


"""
random.seed(1)


def check_img(i, img_root):
    # i['file_name'] = i['file_name'].split('/')[-1]
    try:
        iimage = utils.read_image(opj(img_root, i["file_name"]), format='RGB')
        utils.check_image_size(i, iimage)

    except Exception as e:
        print("BAD D2 IMG", i)
        return i['id']

    return None

def fix_img_size(i, img_root):
    try:
        if not "file_name" in i:
            i["file_name"] = i["coco_url"].split("/")[-1]
        img = Image.open(opj(img_root, i['file_name']))
        w, h = img.size
        if i['width'] != w or i['height'] != h:
            print("Found image {} with wrong size.\n".format(i['id']))
            i['width'] = w
            i['height'] = h

        return i
    except Exception as e:
        print("BAD IMG", i, e)
        return None


def fix_data(img_root, data):
    # first fix sizes
    num_imgs = len(data['images'])
    
    data['images'] = Parallel(n_jobs=15, backend='threading')(delayed(fix_img_size)(i, img_root) for i in tqdm(data['images']))
    data['images'] = [i for i in data['images'] if i is not None]
    print("First stage image fixing go from {} to {}".format(num_imgs, len(data['images'])))

    bad_ids = Parallel(n_jobs=15, backend='threading')(delayed(check_img)(i, img_root) for i in tqdm(data['images']))
    bad_ids = [x for x in set(bad_ids) if x is not None]
    print("Found {} bad images with D2 checking".format(len(bad_ids)))
    data['images'] = [d for d in data['images'] if d['id'] not in bad_ids]
    print("Images go from {} to {}".format(num_imgs, len(data['images'])))

    prev_anno_size = len(data['annotations'])
    valid_imgs = {i['id'] for i in data['images']}
    data['annotations'] = [d for d in data['annotations'] if d['image_id'] in valid_imgs]
    print("Anno go from {} to {} after fixing.".format(prev_anno_size, len(data['annotations'])))
    return data

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

# def download_img(dw, save_dir, dataset_name):
#     res = True
#     try:
#         url = quote(dw['url'], safe=string.printable).replace(" ", "%20")
#         img_url = dw['url'].rsplit('/', 1)[-1]
#         save_img = opj(save_dir, 'images', dataset_name, 'raw')
#         path_mkdir(save_img)
#         urllib.request.urlretrieve(url, opj(save_img, img_url))
#
#     except Exception as e:
#         res = False
#         print(f"{url} cannot be downloaded, save error msg to error.txt")
#
#     return (res, {"url": url, "save_name": img_url})

def download_img(dw, save_dir, dataset_name):
    res = True
    try:
        url=dw['url']
        # url = quote(dw['url'], safe=string.printable).replace(" ", "%20")
        img=Image.open(urllib.request.urlopen(url)).covert("RGB")
        img_name=url.split("/")[-1]
        save_img = opj(save_dir, 'images', dataset_name, 'raw')
        path_mkdir(save_img)
        img.save(opj(save_img,img_name))
    except Exception as e:
        res = False
        print(f"{url} cannot be downloaded, save error msg to error.txt")

    return (res, {"url": url, "save_name": img_name})






def annotations_extraction(datalist, category_map, save_dir, dataset_name):
    category_map_num = copy.deepcopy(category_map)
    for k in category_map_num.keys():
        category_map_num[k] = 0
    images = []
    annotations = []
    im_id = 0

    # res = Parallel(n_jobs=30, backend='threading')(delayed(download_img)(dw, save_dir, dataset_name) for dw in tqdm(datalist))
    [download_img(dw, save_dir, dataset_name) for dw in tqdm(datalist)]
    for i in res:
        if i[0] == False: 
            with open(opj(save_dir, dataset_name, "error.txt"), "a") as f:
                    json.dump(res, f, ensure_ascii=False)
                    f.write("\n")
        

    for one_res, d in tzip(res, datalist):
        is_vaild = one_res[0]
        if not is_vaild:
            continue
        raw_img_file = d['url'].rsplit('/', 1)[-1]
        img_dir = opj(save_dir, 'images', dataset_name, 'raw')
        try:
            img = Image.open(opj(img_dir, raw_img_file)).convert('RGB')
        except IOError:
            # if not img: # 如果文件不存在，则删除下载的文件
            os.remove(opj(img_dir, raw_img_file))
            continue

        uid = str(uuid.uuid4())
        file_n = opj('images', dataset_name, 'raw', uid + '.' + d['url'].rsplit('.', 1)[-1])
        os.rename(opj(img_dir, raw_img_file), opj(save_dir, file_n))
        file_img = opj(save_dir, file_n)

        img_width = img.size[0] 
        img_height = img.size[1] 

        image = {
            "license": 0,
            "file_name": file_n,
            "coco_url": d['url'],
            "height": img_height,
            "width": img_width,
            "id": uid
        }
        
        img.save(file_img)
        img.close()

        if not os.path.isfile(file_img):
            print(f"{file_img} does not exist, check your local path!")
            import pdb;
            pdb.set_trace()

        for obj in d['obj']:
            category_map_num[obj['object']] += 1
            bbox = normal_bbox(obj['bbox'])
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            xmin, ymin = float(bbox[0]), float(bbox[1])
            xmax, ymax = float(bbox[2]), float(bbox[3])
            if xmin > xmax:
                continue
            if ymin > ymax:
                continue
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if ymax > img_height:
                ymax = img_height
            if xmax > img_width:
                xmax = img_width
            annotation = {
                "segmentation": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                "image_id": uid,
                "bbox": [x, y, w, h],  # xywh
                "category_id": category_map[obj['object']],
                "id": im_id,
                "iscrowd":0,
                "area": w * h
            }
            annotations.append(annotation)
            im_id += 1
        images.append(image)
        

    
    return images, annotations, category_map_num

def path_mkdir(path):
    os.makedirs(path, exist_ok=True)


def convert_studio_to_coco(studio_jsonfile, save_dir, exclude_classes, dataset_name):
    '''
    studio_jsonfile : studio_json路径
    save_dir: save images and annotations路径
    '''
    data = json.load(open(studio_jsonfile))

    category_set = set()
    category_map = {}
    cnt_null = 0
    exclude_classes = exclude_classes.split(',')
    if isinstance(data, dict):
        for file in data['data'][0]['fileList']:
            new_obj = []
            for obj in file['obj']:
                if len(obj['object']) == 0:
                    cnt_null += 1
                    print(f"{obj} does not have valid object annotation, skipping this object...")
                    continue
                if len(exclude_classes) > 0 and obj['object'] in exclude_classes:
                    print(f"{obj['object']} in {exclude_classes}, skipping this object..")
                    continue
                category_set.add(obj['object'])
                new_obj.append(obj)
            file['obj'] = new_obj

    elif isinstance(data, list):
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
                category_set.add(obj['object'])
                new_obj.append(obj)
            file['obj'] = new_obj
    category_list = list(category_set)
    print(category_list)
    print(f"{cnt_null} files have invalid(empty) object annotation.")
    for i in range(len(category_list)):
        category_map[category_list[i]] = i

    if isinstance(data, dict):
        fileList = data['data'][0]['fileList']
    elif isinstance(data, list):
        fileList = data

    random.shuffle(fileList)
    images, annotations, category_map_num = annotations_extraction(fileList, category_map, save_dir, dataset_name)

    category_list = list(category_map.keys())
    categories = [{'supercategory': category_list[i], 'id': category_map[category_list[i]], 'name': category_list[i]}
                  for i in range(len(category_list))]
    print('category_map_num', category_map_num)


    D = {
        "info": {},
        "licenses": {},
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    coco_data = D
    check_data = fix_data(opj(save_dir, dataset_name), coco_data)


    return check_data


def studio_to_coco(args):
    coco_data = convert_studio_to_coco(args.studio_json_file, args.save_dir, args.exclude_classes, args.dataset_name)
    save_annos = opj(args.save_dir, 'annotations', args.dataset_name, 'format_coco')
    path_mkdir(save_annos)
    with open(opj(save_annos, 'split_all.json'), 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start convert.')
    parser.add_argument('--studio_json_file', type=str, help='studio json file path')
    parser.add_argument('--exclude_classes', type=str, help='class names to exclude, sep by ","', default='')
    parser.add_argument('--save_dir', type=str, help='save images and annotations path')
    parser.add_argument('--dataset_name', type=str, help='dataset name')
    args = parser.parse_args()
    args.studio_json_file="/home/lq/projects/Detic/datasets/mts_ch123/mts_0215.json"
    args.dataset_name="mts_v1"
    args.save_dir="mts_data"
    studio_to_coco(args)
