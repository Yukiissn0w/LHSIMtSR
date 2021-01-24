'''
torch=0.31用のテスト
'''
import argparse
import pickle
import os
import csv

import numpy as np
from tqdm import trange
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str, default='./imp_estimater_dataset2.csv')
parser.add_argument('--name', type=str, default='imp_classfier_torch_test')
parser.add_argument('--save_path', type=str, default='cp/classifier_imp')
parser.add_argument('--gpu', type=str, default='2')
parser.add_argument('--input_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--pre_trained', action='store_true')
parser.add_argument('--model', type=str, choices=['vgg11', 'resnet101', 'pretrained_resnet101'], default='resnet101')
parser.add_argument('--pt_path', type=str, default='imp_classifiee/no_exist.pt')
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

from dataset2 import ClassImageLoader
from sampler2 import ImbalancedDatasetSampler

save_dir = os.path.join(args.save_path, args.name)
os.makedirs(save_dir, exist_ok=True)

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

    test_paths=[]
    with open(args.csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['data_for']=='test': test_paths.append(row['path'])
    sep_data = {'test':test_paths}

    loader = lambda s: ClassImageLoader(paths=sep_data[s], transform=transform[s])
    
    test_set = loader('test')

    test_loader = torch.utils.data.DataLoader(
            test_set,
            # sampler=ImbalancedDatasetSampler(test_set),
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=8)

    num_classes = test_set.num_classes

    print("num_classes : {}".format(num_classes))

    if args.model=='resnet101':    
        from my_model import my_resnet101
        model = my_resnet101(2)
        model_name = "resnet101"

    elif args.model=='vgg11':
        from my_model import my_vgg11
        model = my_vgg11(2)
        model_name = "vgg11"

    elif args.model=='pretrained_resnet101':
        model = models.resnet101(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2) # >> line205でerror
    

    # classifier_path = './cp/classifier_imp/imp_classifier_temp1/resnet101_epoch95_step20640.pt'
    classifier_path = './cp/classifier_imp/{}'.format(args.pt_path)
    classifier = model.cuda()
    classifier.load_state_dict(torch.load(classifier_path))
    classifier.eval()

    # torch <= 0.31
    from torch.autograd import Variable

    # print("test_set.paths:{}".format(test_set.paths))
    for j, data_ in enumerate(test_loader):
        inputs_, labels_ = (Variable(d.cuda()) for d in data_)
        predicted = classifier(inputs_).detach()
        acc_ = accuracy(predicted.data, labels_.data)

        # print("inputs_ : {}\n labels_ : {}\n predicted : {}".format(inputs_, labels_, predicted))
        # print("labels_ : {}\n predicted : {}".format(labels_, predicted))
        print("acc_ : {}".format(acc_))
        input()
