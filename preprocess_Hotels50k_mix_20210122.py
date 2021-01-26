import argparse
import json
import imageio
import os
import scipy.io
import scipy.misc
import sys
import time
import warnings
import numpy as np
from shutil import copyfile
import glob
import csv
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageDraw
import cv2

# seg画像からObjectClassMasks, ObjectInstanceMasks, objectsを求めて返す関数
def loadHotels50k(file, data_for):
    # 元画像とseg画像のnameの対応が統一されている必要がある
    segfile = file.replace('/{}_image'.format(data_for),'/{}_label'.format(data_for)).replace('.png', '_seg.png')
    seg = scipy.misc.imread(segfile)
    
    R, G, B = seg[:,:,0], seg[:,:,1], seg[:,:,2]

    ObjectClassMasks = (R.astype('uint16') / 10) * 256 + G.astype('uint16')
    _, Minstances_hat = np.unique(B, return_inverse=True)
    ObjectInstanceMasks = np.reshape(Minstances_hat, B.shape)
    
    attfile = segfile.replace('/{}_label'.format(data_for),'/{}_atr'.format(data_for)).replace('_seg.png', '_atr.txt')

    with open(attfile, 'r') as f:
        atts = f.readlines()
    C = []
    for att in atts:
        C.append(att.split('# '))
    
    instance = [int(c[0]) for c in C]
    names = [c[1].strip() for c in C]
    
    objects = {}
    objects['instancendx'] = []
    objects['class'] = []
    objects['corrected_raw_name'] = []
    objects['iscrop'] = []
    objects['listattributes'] = []
    for i in range(len(instance)):
        objects['instancendx'].append(instance[i])
        objects['class'].append(names[i])
        objects['corrected_raw_name'].append(names[i])
        objects['iscrop'].append(names[i])
        objects['listattributes'].append(names[i])
    
    return ObjectClassMasks, ObjectInstanceMasks, objects

def main(data_for):
    ade_name = 'Hotels50k_imp_20210120_copy_0125'  # 対象のフォルダ
    src_dir = './datasets'
    save_dir = os.path.join(src_dir, ade_name)
    bbox_train_dir = 'train_bbox'
    bbox_val_dir = 'val_bbox'
    img_train_dir = 'train_img'
    img_val_dir = 'val_img'
    label_train_dir = 'train_label'
    label_val_dir = 'val_label'
    inst_train_dir = 'train_inst'
    inst_val_dir = 'val_inst'
    bbox_suf = '_gtFine_instanceIds.json'
    img_suf = '_leftImg8bit.png'
    label_suf = '_gtFine_labelIds.png'
    inst_suf = '_gtFine_instanceIds.png'
    mat_file = 'color150.mat'    # ade20kベースだが一応学習に使用したモデルに付属していたものを使用する

    dir_names = [bbox_train_dir, bbox_val_dir, img_train_dir, img_val_dir,
        label_train_dir, label_val_dir, inst_train_dir, inst_val_dir]

    for dir_name in dir_names:
        if not os.path.exists(os.path.join(save_dir, dir_name)):
            os.makedirs(os.path.join(save_dir, dir_name))
    
    sorted_50_id = [8, 9, 19, 20, 24, 25, 34, 37, 58]
    sorted_50_name = ['bed', 'windowpane;window', 'curtain;drape;drapery;mantle;pall',
                    'chair', 'sofa;couch;lounge', 'shelf', 'desk', 'lamp', 'pillow']
    sorted_50 = [5227.4, 6118.0, 6579.0, 5292.4, 383.6, 6535.0, 511.0, 5989.4, 235.0]
    # read mat file
    index = scipy.io.loadmat(os.path.join('./datasets/', ade_name, mat_file))
    # read csv file
    csv_file = 'object150_info.csv'
    pd_csv = pd.read_csv(os.path.join('./datasets/', ade_name, csv_file), encoding="ms932", sep=",")
    sorted_colors = []
    for i in range(150):
        for j, obj_name in enumerate(sorted_50_name):
            if obj_name==pd_csv.values[i,5]:
                sorted_colors.append(index['colors'][i])
    sorted_colors = np.array(sorted_colors)

    # Progress bar
    width = 55
    sys.stdout.write("Progress: [%s]" % (" " * width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (width+1))

    # for i, id_ in enumerate(ids):
    folder = './datasets/{}/images/{}_image'.format(ade_name, data_for)
    file_paths = glob.glob(os.path.join(folder, '*.png')) 
    # print(len(file_paths))

    for i, file_path in enumerate(tqdm(file_paths)): 
    
        # original用
        ori_fn = file_path.split('/')[-1].replace('.png', '')

        Om, Oi, objects = loadHotels50k(file_path, data_for)
        
        r, c = Oi.shape # print(r,c)  # >> 424, 640
        label_map = np.zeros((r, c))
        fine_label = Om

        real_img = Image.open(file_path)
        draw = ImageDraw.Draw(real_img)
        segfile = file_path.replace('/{}_image'.format(data_for),'/{}_label'.format(data_for)).replace('.png', '_seg.png')
        seg = scipy.misc.imread(segfile)
        
        h,w,cha = seg.shape
        seg_colors = seg.reshape(h*w,cha)
        uniq_colors = np.unique(seg_colors, axis=0)

        for j, sorted_id in enumerate(sorted_50):
            label_map[fine_label == sorted_id] = 1
            fine_label[fine_label == sorted_id] = j + 1
        fine_label[label_map == 0] = 0
        input()
        if max(fine_label)==0: continue

        bbox_data = {}
        bbox_data['imgHeight'] = int(r)
        bbox_data['imgWidth'] = int(c)
        bbox_data['objects'] = {}

        object_colors = []
        object_names = []
        for j, sorted_color in enumerate(sorted_colors):
            for k, uniq_color in enumerate(uniq_colors):
                if np.all(uniq_color==sorted_color): 
                    object_colors.append(sorted_color)
                    object_names.append(sorted_50_name[j])
        
        for j, object_color in enumerate(object_colors):
            row, col = np.where(((seg[:,:, 0] == object_color[0]) & (seg[:,:, 1] == object_color[1]) & (seg[:,:, 2] == object_color[2])))

            x1, y1 = int(min(col) + 1), int(min(row) + 1)
            x2, y2 = int(max(col) + 1), int(max(row) + 1)

            if (x2-x1)*(y2-y1)<h*w*(0.01): continue
            w, h = x2 - x1, y2 - y1
            margin_x = int(max(round(w / 100), 1))
            margin_y = int(max(round(h / 100), 1))
            x1, y1 = max(x1 - margin_x, 1), max(y1 - margin_y, 1)
            x2, y2 = min(x2 + margin_x, seg.shape[1]), int(min(y2 + margin_y, seg.shape[0]))
            red,green,blue = object_color
            red,green,blue = int(red), int(green), int(blue)
            bbox_data['objects']['{},{},{}'.format(red,green,blue)] = {
                'bbox': [x1, y1, x2, y2],
                'cls': j+1
            }

            # bboxを描画したデバッグ用画像の作成
            # real_img = Image.open(file_path)
            # draw = ImageDraw.Draw(real_img)
            # draw.rectangle([x1, y1, x2, y2], outline=(red,green,blue))
            # print("bbox: {}, cls: {}, color: {}".format([x1, y1, x2, y2], j+1, [red,green,blue]))
            # real_img.show()
            # input()
            # real_img.save('./datasets/bbox_preprocess_test/bbox{}_{}-{}.jpg'.format(i, j, object_names[j]), quality=95)

        prefix = '{}_fn_{}'.format('{0:05}'.format(i+1), ori_fn)

        if data_for=='train':
            bbox_file = os.path.join(save_dir, bbox_train_dir,
                prefix + bbox_suf)
            img_file = os.path.join(save_dir, img_train_dir,
                prefix + img_suf)
            lbl_path = os.path.join(save_dir, label_train_dir,
                prefix + label_suf)
            ist_path = os.path.join(save_dir, inst_train_dir,
                prefix + inst_suf)
        elif data_for=='val':
            bbox_file = os.path.join(save_dir, bbox_val_dir,
                prefix + bbox_suf)
            img_file = os.path.join(save_dir, img_val_dir,
                prefix + img_suf)
            lbl_path = os.path.join(save_dir, label_val_dir,
                prefix + label_suf)
            ist_path = os.path.join(save_dir, inst_val_dir,
                prefix + inst_suf)
        else: error_input=input("--- error line 207 ---")

        with open(bbox_file, 'w') as outfile:
            json.dump(bbox_data, outfile)
        copyfile(file_path, img_file)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imageio.imwrite(lbl_path, fine_label.astype('uint8'))
            imageio.imwrite(ist_path, Oi.astype('uint8'))

        if (i+1) % 100 == 0:
            sys.stdout.write("-")
            sys.stdout.flush()

    sys.stdout.write("\n")

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_for', type=str, choices=['train', 'val'])
    # args = parser.parse_args()
    # main(args.data_for)
    
    main('val')
    
