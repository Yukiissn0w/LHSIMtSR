import scipy.io
import scipy.misc
import csv
import pandas as pd
import glob
import os
import numpy as np
from tqdm import tqdm
from pprint import pprint
from joblib import Parallel, delayed

def process(img_file, read_dir):
    # read image
    seg = scipy.misc.imread(img_file)
    # axis指定でユニークな行の抽出
    h,w,c = seg.shape
    uniq_colors = seg.reshape(h*w,c)
    uniq_colors = np.unique(uniq_colors, axis=0)

    # atr.txtを作成
    atr_file = img_file.split('/')[-1].replace('_seg.png', '_atr.txt')
    # print('uniq_colors : len()={}'.format(len(uniq_colors)))
    with open(os.path.join(read_dir+'/images/train_atr',atr_file), 'w', newline='', encoding="utf-8") as w:
        writer = csv.writer(w)
        for i in range(len(uniq_colors)):
            for j in range(len(index['colors'])):
                if (uniq_colors[i]==index['colors'][j]).all():
                    # print('{} : {}'.format(uniq_colors[i], pd_csv.values[j,5].replace(';',', ')))
                    rgb_str = '{};{};{}'.format(uniq_colors[i][0], uniq_colors[i][1], uniq_colors[i][2])
                    writer.writerow(['{} # {} # {}'.format(str(i).zfill(3), pd_csv.values[j,5], rgb_str)])

if __name__=='__main__':
    # read_dir = './datasets/Hotels50k_mix'
    read_dir = './datasets/Hotels50k_mix_mini_20201209'
    mat_file = 'color150.mat'
    csv_file = 'object150_info.csv'
    os.makedirs(os.path.join(read_dir+'/images/train_atr'), exist_ok=True)  

    # read mat file
    index = scipy.io.loadmat(os.path.join(read_dir, mat_file))

    # read csv file
    pd_csv = pd.read_csv(os.path.join(read_dir,csv_file), encoding="ms932", sep=",")

    file_paths = glob.glob(os.path.join(read_dir+'/images/train_label', '*_seg.png'))
    processed = Parallel(n_jobs=8, verbose=2)([delayed(process)(file_path, 
                read_dir) for file_path in file_paths])
        
