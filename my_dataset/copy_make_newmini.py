import os
import argparse
import csv
import shutil
import glob
from tqdm import tqdm

def copy_make_newmini(read_dir, save_dir, negaposi):
    cnt = 0
    spe_cnt = 0
    real_read_dir = os.path.join(read_dir, '{}tive_real_image_20201208'.format(negaposi))
    seg_read_dir = os.path.join(read_dir, '{}tive_seg_image_20201201'.format(negaposi))
    real_save_dir = os.path.join(save_dir, 'train_image')
    os.makedirs(real_save_dir, exist_ok=True)
    seg_save_dir = os.path.join(save_dir, 'train_label')
    os.makedirs(seg_save_dir, exist_ok=True)
    real_file_paths = glob.glob(os.path.join(real_read_dir, '*.png'))
    # seg_file_paths = glob.glob(os.path.join(read_seg_dir, '*.png'))

    for i, file in tqdm(enumerate(real_file_paths)):
        if negaposi=='nega':
            if int(file.split('/')[-1].split('_')[0])<35:
                real_save_path = os.path.join(real_save_dir,file.split('/')[-1])
                seg_save_path = os.path.join(seg_save_dir,file.split('/')[-1].replace('.png', '_seg.png'))
                
                # print(real_save_path)
                # print(seg_save_path)
                # print(file)
                # print(file.replace('_real_image_20201208', '_seg_image_20201201').replace('.png', '_seg.png'))
                # input()
                # continue

                shutil.copy(file, real_save_path)
                shutil.copy(file.replace('_real_image_20201208', '_seg_image_20201201').replace('.png', '_seg.png'), 
                            seg_save_path)
                cnt+=1
        else:
            if int(file.split('/')[-1].split('_')[0])>47:
                real_save_path = os.path.join(real_save_dir,file.split('/')[-1])
                seg_save_path = os.path.join(seg_save_dir,file.split('/')[-1].replace('.png', '_seg.png'))
                shutil.copy(file, real_save_path)
                shutil.copy(file.replace('_real_image_20201208', '_seg_image_20201201').replace('.png', '_seg.png'), 
                            seg_save_path)
                cnt+=1
            if cnt<1870 and spe_cnt<962:
                if int(file.split('/')[-1].split('_')[0])==47:
                    real_save_path = os.path.join(real_save_dir,file.split('/')[-1])
                    seg_save_path = os.path.join(seg_save_dir,file.split('/')[-1].replace('.png', '_seg.png'))
                    shutil.copy(file, real_save_path)
                    shutil.copy(file.replace('_real_image_20201208', '_seg_image_20201201').replace('.png', '_seg.png'), 
                                seg_save_path)
                    spe_cnt+=1
    print('cnt:{}, spe_cnt:{}'.format(cnt, spe_cnt))

if __name__=='__main__':
    copy_make_newmini('../negaposi_dataset/',
                    './datasets/Hotels50k_mix_mini_20201209/images/',
                    'nega')
    copy_make_newmini('../negaposi_dataset/',
                    './datasets/Hotels50k_mix_mini_20201209/images/',
                    'posi')
