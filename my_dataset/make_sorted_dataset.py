import glob
import os
import numpy as np
import scipy.io
import scipy.misc
import pandas as pd
from tqdm import tqdm
import shutil
from joblib import Parallel, delayed

def process(file_path, sorted_colors, negaposi, real_read_dir, seg_read_dir, real_save_dir, seg_save_dir):
        seg = scipy.misc.imread(file_path)
        h,w,c = seg.shape
        uniq_colors = seg.reshape(h*w,c)
        uniq_colors = np.unique(uniq_colors, axis=0)
        # print(uniq_colors)
        # print(np.any(np.all(np.array([204, 5, 255])==uniq_colors, axis=1)))
        # input()
        # continue
        for j, uniq_color in enumerate(uniq_colors):
            # sorted_colorsの中にuniq_colorがあればTrueが返る
            if np.any(np.all(sorted_colors==uniq_color, axis=1)):
                shutil.copy(file_path, seg_save_dir)
                shutil.copy(os.path.join(real_read_dir, file_path.split('/')[-1].replace('_seg.png', '.png')),
                            real_save_dir)
                # print('achieved')
                # input()
                break

def make_sorted_dataset(sorted_colors, negaposi, real_save_dir, seg_save_dir):
    real_read_dir = '../negaposi_dataset/ori_20201208/{}tive_real_image_20201208'.format(negaposi)
    seg_read_dir = '../negaposi_dataset/ori_20201208/{}tive_seg_image_20201201'.format(negaposi)
    file_paths = glob.glob(os.path.join(seg_read_dir, '*.png'))

    print('len(file_paths):{}'.format(len(file_paths)))
    processed = Parallel(n_jobs=8, verbose=2)([delayed(process)(file_path, 
        sorted_colors, negaposi, real_read_dir, seg_read_dir, real_save_dir,
        seg_save_dir) for file_path in file_paths])


# def make_sorted_dataset(sorted_colors, negaposi, real_save_dir, seg_save_dir):
#     real_read_dir = '../negaposi_dataset/ori_20201208/{}tive_real_image_20201208'.format(negaposi)
#     seg_read_dir = '../negaposi_dataset/ori_20201208/{}tive_seg_image_20201201'.format(negaposi)
#     file_paths = glob.glob(os.path.join(seg_read_dir, '*.png'))

#     print('len(file_paths):{}'.format(len(file_paths)))
#     for i, file_path in tqdm(enumerate(file_paths)):
#         seg = scipy.misc.imread(file_path)
#         h,w,c = seg.shape
#         uniq_colors = seg.reshape(h*w,c)
#         uniq_colors = np.unique(uniq_colors, axis=0)
#         # print(uniq_colors)
#         # print(np.any(np.all(np.array([204, 5, 255])==uniq_colors, axis=1)))
#         # input()
#         # continue
#         for j, uniq_color in enumerate(uniq_colors):
#             # sorted_colorsの中にuniq_colorがあればTrueが返る
#             if np.any(np.all(sorted_colors==uniq_color, axis=1)):
#                 shutil.copy(file_path, seg_save_dir)
#                 shutil.copy(os.path.join(real_read_dir, file_path.split('/')[-1].replace('_seg.png', '.png')),
#                             real_save_dir)
#                 # print('achieved')
#                 # input()
#                 break

if __name__=='__main__':
    read_dir = './datasets/Hotels50k_mix_mini_20201209'
    mat_file = 'color150.mat'
    csv_file = 'object150_info.csv'
    # read mat file
    index = scipy.io.loadmat(os.path.join(read_dir, mat_file))
    # read csv file
    pd_csv = pd.read_csv(os.path.join(read_dir,csv_file), encoding="ms932", sep=",")

    # print(index['colors'])
    # print(index['colors'][7])   # [204, 5, 255]
    # print(pd_csv.values[7,5]) # bed（9行5列目）
    # exit()
    # pd_csv.values[i,j]とindex['colors'][i]は対応している

    # sorted_id = [8, 9, 19, 20, 24, 25, 34, 37, 58]
    sorted_name = ['bed', 'windowpane;window', 'curtain;drape;drapery;mantle;pall',
                    'chair', 'sofa;couch;lounge', 'shelf', 'desk', 'lamp', 'pillow']
    sorted_colors = []
    for i in range(150):
        for j, obj_name in enumerate(sorted_name):
            if obj_name==pd_csv.values[i,5]:
                sorted_colors.append(index['colors'][i])
    # print(type(sorted_colors[0]))   # ndarray
    sorted_colors = np.array(sorted_colors)

    real_save_dir = '../negaposi_dataset/sorted_dataset_20201212/train_image/'
    os.makedirs(real_save_dir, exist_ok=True)
    seg_save_dir = '../negaposi_dataset/sorted_dataset_20201212/train_label/'
    os.makedirs(seg_save_dir, exist_ok=True)

    make_sorted_dataset(sorted_colors, 'nega', real_save_dir, seg_save_dir)
    make_sorted_dataset(sorted_colors, 'posi', real_save_dir, seg_save_dir)
