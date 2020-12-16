import json
import os
import glob

def rm_empty_file(read_dir, traval):
    bbox_suf = '_gtFine_instanceIds.json'
    img_suf = '_leftImg8bit.png'
    label_suf = '_gtFine_labelIds.png'
    inst_suf = '_gtFine_instanceIds.png'
    main_dir = './datasets/Hotels50k_mix_mini_20201212/'

    json_paths = glob.glob(os.path.join(read_dir, '*.json'))
    cnt=0
    # print('all : {}'.format(len(json_paths)))
    for i, json_path in enumerate(json_paths):
        json_open = open(json_path, 'r')
        json_load = json.load(json_open)
        if len(json_load['objects'])==0:
            # print('path : {}'.format(json_path))
            # print('img : {}'.format(json_path.replace('{}_bbox'.format(traval), '{}_img'.format(traval)).replace(bbox_suf, img_suf)))
            cnt+=1
            # os.remove(json_path)
            # os.remove(json_path.replace('{}_bbox'.format(traval), '{}_img'.format(traval)).replace(bbox_suf, img_suf))
            # os.remove(json_path.replace('{}_bbox'.format(traval), '{}_inst'.format(traval)).replace(bbox_suf, inst_suf))
            # os.remove(json_path.replace('{}_bbox'.format(traval), '{}_label'.format(traval)).replace(bbox_suf, label_suf))
            # break
    print('cnt:{}'.format(cnt))

if __name__ == "__main__":
    rm_empty_file('./datasets/Hotels50k_mix_mini_20201212/train_bbox', 'train')
    rm_empty_file('./datasets/Hotels50k_mix_mini_20201212/val_bbox', 'val')
