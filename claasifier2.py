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
parser.add_argument('--csv_path', type=str, default='./imp_estimater_dataset2.csv')
# parser.add_argument('--name', type=str, default='i2w_classifier')
# parser.add_argument('--save_path', type=str, default='cp/classifier_i2w')
parser.add_argument('--name', type=str, default='imp_classfier_torch_test')
parser.add_argument('--save_path', type=str, default='cp/classifier_imp')
parser.add_argument('--gpu', type=str, default='2')
# parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--input_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
# parser.add_argument('--mode', type=str, default='T')
parser.add_argument('--pre_trained', action='store_true')
parser.add_argument('--amp', action='store_true')
parser.add_argument('--model', type=str, choices=['vgg11', 'resnet101', 'pretrained_resnet101'], default='resnet101')
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

if args.amp:
    from apex import amp, optimizers

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
    # # load data
    # with open(args.pkl_path, 'rb') as f:
    #     sep_data = pickle.load(f)
    # # 学習用とテスト用どちらの推定器を作成するかmode選択
    # if args.mode == 'E': 
    #     sep_data['train'] = sep_data['exp']
    # print('{} train data were loaded'.format(len(sep_data['train'])))

    print(args)
    # exit()

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
    val_transform = transforms.Compose([
        transforms.Resize((args.input_size,) * 2),
        transforms.ToTensor(),
        # torch >= 1.7
        # transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform = {'train': train_transform, 'val': val_transform}

    train_paths = []
    val_paths = []
    with open(args.csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['data_for']=='train': train_paths.append(row['path'])
            elif row['data_for']=='val': val_paths.append(row['path'])
    sep_data = {'train':train_paths, 'val':val_paths}

    # loader = lambda s: ClassImageLoader(paths=sep_data[s], transform=transform[s])
    loader = lambda s: ClassImageLoader(paths=sep_data[s], transform=transform[s])
    
    train_set = loader('train')
    val_set = loader('val')

    # print("type(train_set):{}".format(type(train_set))) # >> <class 'dataset2.ClassImageLoader'>

    train_loader = torch.utils.data.DataLoader(
            train_set,
            # sampler=ImbalancedDatasetSampler(train_set),
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=8)

    val_loader = torch.utils.data.DataLoader(
            val_set,
            # sampler=ImbalancedDatasetSampler(val_set),
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=8)

    num_classes = train_set.num_classes
    # print(num_classes) # >> 2

    # modify exist resnet101 model
    # if not args.pre_trained:
    #     model = models.resnet101(pretrained=False, num_classes=num_classes)
    # else:
    #     model = models.resnet101(pretrained=True)
    #     num_features = model.fc.in_features
    #     model.fc = nn.Linear(num_features, num_classes)

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
        
    # model = models.resnet101(num_classes=2)


    # model = models.mobilenet_v2(pretrained=False, num_classes=num_classes)
    
    # print(model)

    # torch >= 1.7
    # model.to('cuda')
    
    # torch >= 0.4
    # device = torch.device('cuda')
    # model.to(device)

    # torch <= 0.31
    model = model.cuda()

    # train setting
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    # criterion2 = nn.BCELoss()
    global_step = 0
    eval_per_iter = 500
    save_per_epoch = 5
    tqdm_iter = trange(args.num_epoch, desc='Training', leave=True)

    comment = '_lr-{}_bs-{}_ne-{}_x{}_name-{}_pre-train-{}_amp-{}'.format(args.lr, args.batch_size, args.num_epoch, args.input_size, args.name, args.pre_trained, args.amp)
    # comment = 'torch031_test_epoch1'
    writer = SummaryWriter(comment=comment)

    if args.amp:
        model, opt = amp.initialize(model, opt, opt_level='O1')

    loss_li = []
    prec_li = []
    # loss_li2 = []

    # out_path = os.path.join(save_dir, 'resnet101_epoch' + '2' + '_step' + '0' + '.pt')
    # torch.save(model, out_path)
    # exit()

    # torch <= 0.31
    from torch.autograd import Variable

    for epoch in tqdm_iter:

        for i, data in enumerate(train_loader, start=0):
            tqdm_iter.set_description('Training [ {} step ]'.format(global_step))
            # inputs, labels = (d.to('cuda') for d in data)
            inputs, labels = (Variable(d.cuda()) for d in data)
            # torch >= 1.7
            # train_set.transform(inputs)
            opt.zero_grad()

            # outputs = model(inputs)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            prec = accuracy(outputs.detach().data, labels.data)

            # loss2 = criterion2(outputs, labels)

            # loss_li.append(loss.item())
            # prec_li.append(prec.item())
            loss_li.append(loss.data[0])
            prec_li.append(prec)

            # loss_li2.append(loss2.data[0])


            if args.amp:
                with amp.scale_loss(loss, opt) as scale_loss:
                    scale_loss.backward()
            else:
                loss.backward()

            opt.step()

            if global_step % eval_per_iter == 0:

                loss_li_ = []
                prec_li_ = []
                for j, data_ in enumerate(val_loader):
                    # with torch.no_grad():
                    #     inputs_, labels_ = (d.to('cuda') for d in data_)
                    #     # torch >= 1.7
                    #     # train_set.transform(inputs_)
                    #     predicted = model(inputs_)
                    #     loss_ = criterion(predicted, labels_)
                    #     prec_ = accuracy(predicted, labels_)
                    #     loss_li_.append(loss_.item())
                    #     prec_li_.append(prec_.item())
                    ## torch <= 0.31
                    inputs_, labels_ = (Variable(d.cuda()) for d in data_)
                    # train_set.transform(inputs_)
                    predicted = model(inputs_).detach()
                    loss_ = criterion(predicted, labels_)
                    prec_ = accuracy(predicted.data, labels_.data)
                    loss_li_.append(loss_.data[0])
                    prec_li_.append(prec_)
                tqdm_iter.set_postfix(OrderedDict(train_prec=np.mean(prec_li), val_prec=np.mean(prec_li_)))
                writer.add_scalars('loss', {
                    'train': np.mean(loss_li),
                    'val': np.mean(loss_li_)
                    }, global_step)
                writer.add_scalars('accuracy', {
                    'train': np.mean(prec_li),
                    'val': np.mean(prec_li_)
                    }, global_step)
                loss_li = []
                prec_li = []

            global_step += 1
            # out_path = os.path.join(save_dir, 'resnet101_epoch' + str(epoch) + '_step' + str(global_step) + '.pt')
            # torch.save(model.state_dict(), out_path)
            # exit()

        if epoch % save_per_epoch == 0:
            # tqdm_iter.set_description('{} iter: Training loss={:.5f} accuracy={:.5f}'.format(
            #     global_step,
            #     np.mean(loss_li),
            #     np.mean(prec_li)
            #     ))
            out_path = os.path.join(save_dir, '{}_epoch'.format(model_name) + str(epoch) + '_step' + str(global_step) + '.pt')
            torch.save(model.state_dict(), out_path)

    print('Done: training')
