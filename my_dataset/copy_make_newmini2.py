import os
import argparse
import csv
import shutil
import glob
from tqdm import tqdm
from joblib import Parallel, delayed

def copy_make_newmini2(read_dir, save_dir, negaposi):
    cnt = 0
    spe_cnt = 0
    real_save_dir = os.path.join(save_dir, 'train_image')
    os.makedirs(real_save_dir, exist_ok=True)
    seg_save_dir = os.path.join(save_dir, 'train_label')
    os.makedirs(seg_save_dir, exist_ok=True)

    real_file_paths = glob.glob(os.path.join(read_dir, '*.png'))
    for i, file in tqdm(enumerate(real_file_paths)):
        if negaposi=='nega':
            if int(file.split('/')[-1].split('_')[0])<35:
                real_save_path = os.path.join(real_save_dir,file.split('/')[-1])
                seg_save_path = os.path.join(seg_save_dir,file.split('/')[-1].replace('.png', '_seg.png'))
                shutil.copy(file, real_save_path)
                shutil.copy(file.replace('/train_image', '/train_label').replace('.png', '_seg.png'), 
                            seg_save_path)
                cnt+=1
        else:
            if int(file.split('/')[-1].split('_')[0])>47:
                real_save_path = os.path.join(real_save_dir,file.split('/')[-1])
                seg_save_path = os.path.join(seg_save_dir,file.split('/')[-1].replace('.png', '_seg.png'))
                shutil.copy(file, real_save_path)
                shutil.copy(file.replace('/train_image', '/train_label').replace('.png', '_seg.png'), 
                            seg_save_path)
                cnt+=1
            elif cnt<1748 and spe_cnt<902:
                if int(file.split('/')[-1].split('_')[0])==47:
                    real_save_path = os.path.join(real_save_dir,file.split('/')[-1])
                    seg_save_path = os.path.join(seg_save_dir,file.split('/')[-1].replace('.png', '_seg.png'))
                    shutil.copy(file, real_save_path)
                    shutil.copy(file.replace('/train_image', '/train_label').replace('.png', '_seg.png'), 
                                seg_save_path)
                    spe_cnt+=1

    print('cnt:{}, spe_cnt:{}'.format(cnt, spe_cnt))

if __name__=='__main__':
    copy_make_newmini2('../negaposi_dataset/sorted_dataset_20201212/train_image',
                    './datasets/Hotels50k_mix_mini_20201212/images/',
                    'nega')
    copy_make_newmini2('../negaposi_dataset/sorted_dataset_20201212/train_image',
                    './datasets/Hotels50k_mix_mini_20201212/images/',
                    'posi')
