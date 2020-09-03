import os
import glob
from shutil import copy2
from PIL import Image
import json
import numpy as np
from tqdm import tqdm

def copy_file(src, src_ext, dst):
    # find all files ends up with ext
    flist = sorted(glob.glob(os.path.join(src, '*', src_ext)))
    for fname in flist:
        copy2(fname, dst)
        # print('copied %s to %s' % (fname, dst))

def construct_box(inst_root, inst_name, cls_name, dst):
    inst_list = sorted(glob.glob(os.path.join(inst_root, '*', inst_name)))
    cls_list = sorted(glob.glob(os.path.join(inst_root, '*', cls_name)))
    print(len(cls_list))
    for inst, cls in tqdm(zip(*(inst_list, cls_list))):
        inst_map = Image.open(inst)
        inst_map = np.array(inst_map, dtype=np.int32)
        cls_map = Image.open(cls)
        cls_map = np.array(cls_map, dtype=np.int32)
        H, W = inst_map.shape
        # get a list of unique instances
        inst_info = {'imgHeight':H, 'imgWidth':W, 'objects':{}}
        inst_ids = np.unique(inst_map)
        for iid in inst_ids: 
            if int(iid) < 1000: # filter out non-instance masks
                continue
            
            ys,xs = np.where(inst_map==iid)
            ymin, ymax, xmin, xmax = \
                    int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
            cls_label = np.median(cls_map[inst_map==iid])
            # print('xmin:{}, type:{}'.format(xmin, type(xmin)))
            # print('ymin:{}, type:{}'.format(ymin, type(ymin)))
            # print('xmax:{}, type:{}'.format(xmax, type(xmax)))
            # print('ymax:{}, type:{}'.format(ymax, type(ymax)))

            inst_info['objects'][str(iid)] = {'bbox': [xmin, ymin, xmax, ymax], 'cls':int(cls_label)}
        # write a file to path
        filename = os.path.splitext(os.path.basename(inst))[0]
        savename = os.path.join(dst, filename + '.json')
        #ch
        # print('inst_info:{}'.format(inst_info))
        # json.dump(inst_info)
        
        with open(savename, 'w') as f:
            json.dump(inst_info, f)
        # print('wrote a bbox summary of %s to %s' % (inst, savename))

# organize image
if __name__ == '__main__':
    folder_name = 'datasets/cityscape/'
    train_img_dst = os.path.join(folder_name, 'train_img')
    train_label_dst = os.path.join(folder_name, 'train_label')
    train_inst_dst = os.path.join(folder_name, 'train_inst')
    train_bbox_dst = os.path.join(folder_name, 'train_bbox')
    val_img_dst = os.path.join(folder_name, 'val_img')
    val_label_dst = os.path.join(folder_name, 'val_label')
    val_inst_dst = os.path.join(folder_name, 'val_inst')
    val_bbox_dst = os.path.join(folder_name, 'val_bbox')

    os.makedirs(train_img_dst, exist_ok=True)
    os.makedirs(train_label_dst, exist_ok=True)
    os.makedirs(train_inst_dst, exist_ok=True)
    os.makedirs(val_img_dst, exist_ok=True)
    os.makedirs(val_label_dst, exist_ok=True)
    os.makedirs(val_inst_dst, exist_ok=True)

    # train_image
    copy_file('datasets/cityscape/leftImg8bit/train',\
            '*_leftImg8bit.png', train_img_dst)
    # train_label
    copy_file('datasets/cityscape/gtFine/train',\
            '*_labelIds.png', train_label_dst)
    # train_inst
    copy_file('datasets/cityscape/gtFine/train',\
            '*_instanceIds.png', train_inst_dst)
    # val_image
    copy_file('datasets/cityscape/leftImg8bit/val',\
            '*_leftImg8bit.png', val_img_dst)
    # val_label
    copy_file('datasets/cityscape/gtFine/val',\
            '*_labelIds.png', val_label_dst)
    # val_inst
    copy_file('datasets/cityscape/gtFine/val',\
            '*_instanceIds.png', val_inst_dst)

    if not os.path.exists(train_bbox_dst):
        os.makedirs(train_bbox_dst)
    if not os.path.exists(val_bbox_dst):
        os.makedirs(val_bbox_dst)
    # wrote a bounding box summary 
    construct_box('datasets/cityscape/gtFine/train',\
            '*_instanceIds.png', '*_labelIds.png', train_bbox_dst)
    construct_box('datasets/cityscape/gtFine/val',\
            '*_instanceIds.png', '*_labelIds.png', val_bbox_dst) 