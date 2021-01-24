import argparse
import pickle
import os
import csv

import numpy as np
from tqdm import trange
from collections import OrderedDict

parser = argparse.ArgumentParser()
# parser.add_argument('--image_root', type=str, default='/mnt/fs2/2018/matsuzaki/dataset_fromnitta/Image/')
# parser.add_argument('--pkl_path', type=str, default='/mnt/fs2/2019/Takamuro/db/i2w/sepalated_data.pkl')
parser.add_argument('--csv_path', type=str, default='./imp_estimater_dataset.csv')
# parser.add_argument('--name', type=str, default='i2w_classifier')
# parser.add_argument('--save_path', type=str, default='cp/classifier_i2w')
parser.add_argument('--name', type=str, default='imp_classifier')
parser.add_argument('--save_path', type=str, default='cp/classifier_imp')
parser.add_argument('--gpu', type=str, default='2')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
# parser.add_argument('--mode', type=str, default='T')
parser.add_argument('--pre_trained', action='store_true')
parser.add_argument('--amp', action='store_true')
parser.add_argument('--pt_path', type=str, default='imp_classifiee/no_exist.pt')
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

if args.amp:
    from apex import amp, optimizers

from torch.utils.tensorboard import SummaryWriter

from dataset import ClassImageLoader
from sampler import ImbalancedDatasetSampler

save_dir = os.path.join(args.save_path, args.name)
os.makedirs(save_dir, exist_ok=True)


def accuracy(outputs, labels):
    out = torch.argmax(outputs, dim=1)
    return torch.eq(out, labels).float().mean()

if __name__ == '__main__':
    # # load data
    # with open(args.pkl_path, 'rb') as f:
    #     sep_data = pickle.load(f)
    # # 学習用とテスト用どちらの推定器を作成するかmode選択
    # if args.mode == 'E': 
    #     sep_data['train'] = sep_data['exp']
    # print('{} train data were loaded'.format(len(sep_data['train'])))

    # torch >= 1.7
    # train_transform = nn.Sequential([
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
                brightness=0.5,
                contrast=0.3,
                saturation=0.3,
                hue=0
            ),
        transforms.ToTensor(),
        # torch >= 1.7
        # transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # torch >= 1.7
    # test_transform = nn.Sequential([
    test_transform = transforms.Compose([
        transforms.Resize((args.input_size,) * 2),
        transforms.ToTensor(),
        # torch >= 1.7
        # transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform = {'train': train_transform, 'test': test_transform}

    train_paths = []
    test_paths = []
    with open(args.csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['data_for']=='train': train_paths.append(row['path'])
            elif row['data_for']=='test': test_paths.append(row['path'])
    sep_data = {'train':train_paths, 'test':test_paths}

    # loader = lambda s: ClassImageLoader(paths=sep_data[s], transform=transform[s])
    loader = lambda s: ClassImageLoader(paths=sep_data[s], transform=transform[s])
    
    train_set = loader('train')
    test_set = loader('test')

    train_loader = torch.utils.data.DataLoader(
            train_set,
            # sampler=ImbalancedDatasetSampler(train_set),
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=8)

    test_loader = torch.utils.data.DataLoader(
            test_set,
            # sampler=ImbalancedDatasetSampler(test_set),
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=8)

    num_classes = train_set.num_classes

    classifier_path = './cp/classifier_imp/{}'.format(args.pt_path)
    classifier = models.resnet101(num_classes=2)
    classifier.load_state_dict(torch.load(classifier_path))
    classifier.to('cuda')
    classifier.eval()

    # print("test_set.paths:{}".format(test_set.paths))
    for j, data_ in enumerate(test_loader):
        with torch.no_grad():
            # print("len(data_) : {}".format(len(data_)))
            inputs_, labels_ = (d.to('cuda') for d in data_)
            # print("len(inputs_):{}, len(labels_):{}".format(len(inputs_), len(labels_))) # >>16

            # torch >= 1.7
            # train_set.transform(inputs_)
            predicted = classifier(inputs_)
            # loss_ = criterion(predicted, labels_)
            acc_ = accuracy(predicted, labels_)

            # print("inputs_ : {}\n labels_ : {}\n predicted : {}".format(inputs_, labels_, predicted))
            # print("labels_ : {}\n predicted : {}".format(labels_, predicted))
            print("acc_ : {}".format(acc_))
            input()
