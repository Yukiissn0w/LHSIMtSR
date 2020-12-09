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

# seg画像からObjectClassMasks, ObjectInstanceMasks, objectsを求めて返す関数
def loadHotels50k_mix(file):
    # 元画像とseg画像のnameの対応が統一されている必要がある
    segfile = file.replace('/train_image/','/train_label/').replace('.jpg', '_seg.png')
    seg = scipy.misc.imread(segfile)
    
    R, G, B = seg[:,:,0], seg[:,:,1], seg[:,:,2]

    ObjectClassMasks = (R.astype('uint16') / 10) * 256 + G.astype('uint16')
    _, Minstances_hat = np.unique(B, return_inverse=True)
    ObjectInstanceMasks = np.reshape(Minstances_hat, B.shape)
    
    attfile = segfile.replace('/train_label/','/train_atr/').replace('_seg.png', '_atr.txt')

    with open(attfile, 'r') as f:
        atts = f.readlines()
    C = []
    for att in atts:
        C.append(att.split('# '))
    
    instance = [int(c[0]) for c in C]
    names = [c[1].strip() for c in C]
    # corrected_raw_name = [c[4].strip() for c in C]
    # partlevel = [int(c[1]) for c in C]
    # ispart = [1 if p > 0 else 0 for p in partlevel]
    # iscrop = [int(c[2]) for c in C]
    # listattributes = [c[5].replace('"','').strip() for c in C]
    
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

if __name__ == '__main__':
    ade_name = 'Hotels50k_mix'  # save_drで使う
    src_dir = 'datasets'
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

    # index = scipy.io.loadmat(os.path.join(save_dir, mat_file))
    # filenames = index['index'][0,0][0][0]
    # folders = index['index'][0,0][1][0]
    # obj_names = index['index'][0,0][6][0]
    # import pprint
    # print('---index---\n{}'.format(index))
    # print('---filenames---\n{}'.format(filenames))
    # print('---folders---\n{}'.format(folders))
    # print('---obj_names---\n{}'.format(obj_names))
    # exit()
    # ids = []
    # bedroom_name = 'images/training/b/bedroom'

    # for i, folder in enumerate(folders):
    #     if '/'.join(folder[0].split('/')[1:]) == bedroom_name:
    #         ids.append(i)

    count_val = 0
    # Top 50 most occurring objects in the datasetのID(ADE20k)
    # sorted_50 = [2978,165,976,2684,1395,447,1735,3055,1869,687,689,774,471,350,491,
    #     1564,2178,236,2932,530,57,2985,1910,978,2243,1451,2982,266,894,2730,2329,
    #     2733,1981,2676,212,1702,724,2473,146,571,1930,206,2046,2850,249,2586,943,480]

    # Progress bar
    width = 55
    sys.stdout.write("Progress: [%s]" % (" " * width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (width+1))

    # for i, id_ in enumerate(ids):
    folder = './datasets/Hotels50k_mix/images/train_image'
    file_paths = glob.glob(os.path.join(folder, '*.jpg')) 
    # print(len(file_paths))

    # read csv file
    read_dir = './datasets/Hotels50k_mix'
    csv_file = 'object150_info.csv'
    pd_csv = pd.read_csv(os.path.join(read_dir,csv_file), encoding="ms932", sep=",")

    for i, file_path in enumerate(tqdm(file_paths)): 
        # folder = os.path.join(*folders[id_][0].split('/')[1:])
        # filename = os.path.join(folder, filenames[id_][0])
        # filename = os.path.join(save_dir, filename)

        Om, Oi, objects = loadHotels50k_mix(file_path)
        
        r, c = Oi.shape
        label_map = np.zeros((r, c))
        fine_label = Om

        # ???
        # for j, sorted_id in enumerate(sorted_50):
        #     label_map[fine_label == sorted_id] = 1
        #     fine_label[fine_label == sorted_id] = j + 1
        # fine_label[label_map == 0] = 0

        bbox_data = {}
        bbox_data['imgHeight'] = int(r)
        bbox_data['imgWidth'] = int(c)
        bbox_data['objects'] = {}

        uniq_ids = np.unique(Oi)

        count = 0
        # Oi(segのB)のunipueがuniq_ids
        for j, uniq_id in enumerate(uniq_ids):
            if uniq_id == 0:
                continue
            cls_ = objects['class'][uniq_id - 1]

            # for k, obj_name in enumerate(obj_names):
            #     if obj_name == cls_:
            #         obj_id = k + 1
            #         break
            for k in range(150):
                if pd_csv.values[k,5] == cls_:
                    name_id = k + 1
                    break

            # name_id = -1
            # for k, s_id in enumerate(sorted_50):
            #     if obj_id == s_id:
            #         name_id = k + 1
            #         break

            # if obj_id in sorted_50:
            # with open(read_path, 'r') as f:
            #     reader = csv.DictReader(f)
            #     for row in reader:
            #         name_id=0
            #         if row['name']==objects['class']:
            #             name_id = int(row['Idx'])
            #     if name_id==0:print("l159:error")
                # if name_id == -1:
                #     print(obj_id)
            if np.sum(Oi == uniq_id) == 0:
                continue
            count += 1
            row, col = np.where(Oi == uniq_id)
            x1, y1 = int(min(col) + 1), int(min(row) + 1)
            x2, y2 = int(max(col) + 1), int(max(row) + 1)
            w, h = x2 - x1, y2 - y1
            margin_x = int(max(round(w / 100), 1))
            margin_y = int(max(round(h / 100), 1))
            x1, y1 = max(x1 - margin_x, 1), max(y1 - margin_y, 1)
            x2, y2 = min(x2 + margin_x, c), min(y2 + margin_y, r)
            bbox_data['objects'][str(uniq_id)] = {
                'bbox': [x1, y1, x2, y2],
                'cls': name_id
        }

        # prefix = 'bedroom_%05d' % (i + 1)
        prefix ='Hotels50k_mix_%05d' % (i+1)
        is_train = False
        if is_train or count_val >= 150:
            bbox_file = os.path.join(save_dir, bbox_train_dir,
                prefix + bbox_suf)
            img_file = os.path.join(save_dir, img_train_dir,
                prefix + img_suf)
            lbl_path = os.path.join(save_dir, label_train_dir,
                prefix + label_suf)
            ist_path = os.path.join(save_dir, inst_train_dir,
                prefix + inst_suf)
        else:
            bbox_file = os.path.join(save_dir, bbox_val_dir,
                prefix + bbox_suf)
            img_file = os.path.join(save_dir, img_val_dir,
                prefix + img_suf)
            lbl_path = os.path.join(save_dir, label_val_dir,
                prefix + label_suf)
            ist_path = os.path.join(save_dir, inst_val_dir,
                prefix + inst_suf)
            count_val += 1
        with open(bbox_file, 'w') as outfile:
            json.dump(bbox_data, outfile)
        copyfile(file_path, img_file)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # imageio.imwrite(lbl_path, fine_label.astype('double') / 255)
            # imageio.imwrite(ist_path, Oi.astype('double') / 255)
            imageio.imwrite(lbl_path, fine_label.astype('uint8'))
            imageio.imwrite(ist_path, Oi.astype('uint8'))

        if (i+1) % 25 == 0:
            sys.stdout.write("-")
            sys.stdout.flush()

    sys.stdout.write("\n")
