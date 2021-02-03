'''
画像変換器による合成画像の印象を推定する
'''
import argparse
import pickle
import os
import csv
from glob import glob

import numpy as np
from tqdm import trange
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='imp_classfier_torch_test')
# parser.add_argument('--save_path', type=str, default='cp/classifier_imp')
parser.add_argument('--gpu', type=str, default='2')
parser.add_argument('--input_size', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--pre_trained', action='store_true')
parser.add_argument('--model', type=str, choices=['vgg11', 'resnet101', 'pretrained_resnet101'], default='resnet101')
parser.add_argument('--pt_path', type=str, default='imp_classifier/no_exist.pt')
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

from dataset2 import ClassImageLoader3
from sampler2 import ImbalancedDatasetSampler

# save_dir = os.path.join(args.save_path, args.name)
# os.makedirs(save_dir, exist_ok=True)

def accuracy(outputs, labels):
    # out = torch.argmax(outputs, dim=1)
    outputs = outputs.cpu().numpy().copy()
    labels = labels.cpu().numpy().copy()
    out = np.array(np.argmax(outputs, axis=1))
    # out = torch.from_numpy(out)
    # return torch.eq(out, labels).float().mean()
    return np.mean((out==labels))

if __name__ == '__main__':
    # test_transform = nn.Sequential([
    test_transform = transforms.Compose([
        transforms.Resize((args.input_size,) * 2),
        transforms.ToTensor(),
        # torch >= 1.7
        # transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform = {'test': test_transform}
    
    if args.model=='resnet101':    
        from my_model import my_resnet101
        classifier = my_resnet101(2)
        model_name = "resnet101"

    elif args.model=='vgg11':
        from my_model import my_vgg11
        classifier = my_vgg11(2)
        model_name = "vgg11"

    elif args.model=='pretrained_resnet101':
        classifier = models.resnet101()
        classifier.avgpool = nn.AvgPool2d(kernel_size=8, stride=1, padding=0, ceil_mode=False, count_include_pad=True)
        num_features = classifier.fc.in_features
        classifier.fc = nn.Linear(num_features, 2)
        model_name = "pretrained_resnet101"

    # classifier_path = './cp/classifier_imp/imp_classifier_temp1/resnet101_epoch95_step20640.pt'
    classifier_path = './cp/classifier_imp/{}'.format(args.pt_path)
    classifier = classifier.cuda()
    classifier.load_state_dict(torch.load(classifier_path))
    classifier.eval()

    # 画像へのパスを直接取得
    # 元画像から元印象と合成画像のパスを取得
    target_list = glob('../LHSIMtSR/results/test_joint_inference_imp_20210131_1/test/images/*_GT_image_canvas.jpg')

    accuracy_li = []
    accuracy_impori_li = []
    accuracy_imprev_li = []

    for i, file_path in enumerate(target_list):
        imp_label = file_path.split('/')[-1].split('_')[1]
        if imp_label=='imp0': imprev_label = 'imp1'
        elif imp_label=='imp1': imprev_label = 'imp0'
        else: input_error = input('line98 >> imp_label is False')
        # print(file_path)
        # print(type(imp_label), imp_label)
        # input()
        file_number = file_path.split('/')[-1].split('_')[0]
        test_impori_paths = glob('../LHSIMtSR/results/test_joint_inference_imp_20210131_1/train/images/{}_predicted_image_patch_cls*.jpg'.format(file_number))
        # print(len(test_impori_paths), test_impori_paths)
        # input()
        test_imprev_paths = glob('../LHSIMtSR/results/test_joint_inference_imp_20210131_1/train/images/{}_predicted_image_patch_imprev_cls*.jpg'.format(file_number))
        # print(len(test_imprev_paths), test_imprev_paths)
        # input()
        # with open(args.csv_path, 'r') as f:
        #     reader = csv.DictReader(f)
        #     for row in reader:
        #         if row['data_for']=='test': test_paths.append(row['path'])
        sep_data = {'test_impori':test_impori_paths, 'test_imprev':test_imprev_paths}

        # loader = lambda s: ClassImageLoader3(paths=sep_data[s], transform=transform['test'], imp_label=imp_label)
        
        test_impori_set = ClassImageLoader3(paths=sep_data['test_impori'], transform=transform['test'], imp_label=imp_label)
        test_imprev_set = ClassImageLoader3(paths=sep_data['test_imprev'], transform=transform['test'], imp_label=imprev_label)

        test_impori_loader = torch.utils.data.DataLoader(
                test_impori_set,
                # sampler=ImbalancedDatasetSampler(test_set),
                batch_size=args.batch_size,
                drop_last=True,
                num_workers=8)
        test_imprev_loader = torch.utils.data.DataLoader(
                test_imprev_set,
                # sampler=ImbalancedDatasetSampler(test_set),
                batch_size=args.batch_size,
                drop_last=True,
                num_workers=8)

        # num_classes = test_impori_set.num_classes
        # print("num_classes : {}".format(num_classes))

        # torch <= 0.31
        from torch.autograd import Variable

        # print("test_impori_set.paths:{}".format(test_impori_set.paths))
        # input()
        every_accuracy_impori_li = []
        every_accuracy_imprev_li = []
        for j, data_ in enumerate(test_impori_loader):
            inputs_, labels_ = (Variable(d.cuda()) for d in data_)
            # print(labels_.data)
            predicted = classifier(inputs_).detach()
            acc_ = accuracy(predicted.data, labels_.data)

            # print("inputs_ : {}\n labels_ : {}\n predicted : {}".format(inputs_, labels_, predicted))
            # print("labels_ : {}\n predicted : {}".format(labels_, predicted))
            # print("acc_ : {}".format(acc_))
            every_accuracy_impori_li.append(acc_)
            accuracy_li.append(acc_)
            accuracy_impori_li.append(acc_)
        # print('every_accuracy_impori_li:{}'.format(sum(every_accuracy_impori_li)/len(every_accuracy_impori_li)))

        for j, data_ in enumerate(test_imprev_loader):
            inputs_, labels_ = (Variable(d.cuda()) for d in data_)
            # print(labels_.data)
            predicted = classifier(inputs_).detach()
            acc_ = accuracy(predicted.data, labels_.data)

            # print("inputs_ : {}\n labels_ : {}\n predicted : {}".format(inputs_, labels_, predicted))
            # print("labels_ : {}\n predicted : {}".format(labels_, predicted))
            # print("acc_ : {}".format(acc_))
            every_accuracy_imprev_li.append(acc_)
            accuracy_li.append(acc_)
            accuracy_imprev_li.append(acc_)
        # print('every_accuracy_imprev_li:{}'.format(sum(every_accuracy_imprev_li)/len(every_accuracy_imprev_li)))
        # input()
    print('accuracy_li:{}'.format(sum(accuracy_li)/len(accuracy_li)))
    print('accuracy_impori_li:{}'.format(sum(accuracy_impori_li)/len(accuracy_impori_li)))
    print('accuracy_imprev_li:{}'.format(sum(accuracy_imprev_li)/len(accuracy_imprev_li)))

