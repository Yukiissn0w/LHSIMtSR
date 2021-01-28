### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import sys
from options.mask2image_train_options import MaskToImageTrainOptions as TrainOptions
opt = TrainOptions().parse(default_args=sys.argv[1:])
# opt = TrainOptions().parse()
print(sys.argv[1:])
# input("line9")
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids[0])
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import time
from collections import OrderedDict
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import torch

iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)

# tensorboard
from tensorboardX import SummaryWriter
# comment = '_lr-{}_bs-{}_ne-{}_x{}_name-{}_pre-train-{}_amp-{}'.format(args.lr, args.batch_size, args.num_epoch, args.input_size, args.name, args.pre_trained, args.amp)
comment = '_{}_ne_{}_x{}_lfea{}_lrec{}_limp{}'.format(opt.name, opt.niter, opt.loadSize, int(opt.lambda_feat), int(opt.lambda_rec), int(opt.lambda_imp))
writer = SummaryWriter(comment=comment)

save_dir = './cp/{}'.format(opt.name)
os.makedirs(save_dir, exist_ok=True)

# # 印象推定器の読み込み
# from my_model import my_resnet101
# classifier = my_resnet101(2)
# # debug用のptパス
# pt_path = 'imp_classifier_9/resnet101_epoch110_step21756.pt'
# classifier_path = '../Weather_UNet_v2/cp/classifier_imp/{}'.format(pt_path)
# classifier.load_state_dict(torch.load(classifier_path))
# classifier = classifier.cuda()
# classifier.eval()

# # pretrained 印象推定器の読み込み input_size=512
# import torchvision.models as models
# import torch.nn as nn
# classifier = models.resnet101()
# classifier.avgpool = nn.AvgPool2d(kernel_size=16, stride=1, padding=0, ceil_mode=False, count_include_pad=True)
# num_features = classifier.fc.in_features
# classifier.fc = nn.Linear(num_features, 2)
# pt_path = 'imp_classifier_10/resnet101_pretrained_epoch20_step4116.pt'
# classifier_path = '../Weather_UNet_v2/cp/classifier_imp/{}'.format(pt_path)
# classifier.load_state_dict(torch.load(classifier_path))
# classifier = classifier.cuda()
# classifier.eval()

# pretrained 印象推定器の読み込み input_size=256
import torchvision.models as models
import torch.nn as nn
classifier = models.resnet101()
classifier.avgpool = nn.AvgPool2d(kernel_size=8, stride=1, padding=0, ceil_mode=False, count_include_pad=True)
num_features = classifier.fc.in_features
classifier.fc = nn.Linear(num_features, 2)
pt_path = '/imp_classifier_11/resnet101_pretrained_epoch15_step3136.pt'
classifier_path = '../Weather_UNet_v2/cp/classifier_imp/{}'.format(pt_path)
classifier.load_state_dict(torch.load(classifier_path))
classifier = classifier.cuda()
classifier.eval()

# tensorboard Loss
loss_D_li = []
loss_D_fake_li = []
loss_D_real_li = []
loss_G_li = []
loss_G_GAN_li = []
loss_G_GAN_feat_li = []
loss_G_VGG_li = []
loss_G_imp_li = []

total_steps = (start_epoch-1) * dataset_size + epoch_iter  
process_start_time = time.time()
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in tqdm(enumerate(dataset, start=epoch_iter)):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == 0

        ############## Forward Pass ######################
        # SH: I chaned to use forward wrapper to pass data as dictionary 
        #losses, generated = model.module.forward_wrapper(data, save_fake) 
        
        # input("line80")

        losses, generated = model(
                label=Variable(data['label']),
                inst=Variable(data['inst']),
                image=Variable(data['image']),
                feat=None,
                mask_in=Variable(data['mask_in']),
                mask_out=Variable(data['mask_out']),
                infer=save_fake,
                classifier=classifier,
                imp_label=Variable(data['imp_label']))

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['G_VGG'] + loss_dict['G_imp']
        #loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['G_VGG'] + loss_dict['G_pixel'] + loss_dict['KL_loss']

        loss_D_li.append(loss_D.data[0])
        loss_D_fake_li.append(loss_dict['D_fake'].data[0]) 
        loss_D_real_li.append(loss_dict['D_real'].data[0]) 
        loss_G_li.append(loss_G.data[0]) 
        loss_G_GAN_li.append(loss_dict['G_GAN'].data[0]) 
        loss_G_GAN_feat_li.append(loss_dict['G_GAN_Feat'].data[0]) 
        loss_G_VGG_li.append(loss_dict['G_VGG'].data[0])
        loss_G_imp_li.append(loss_dict['G_imp'].data[0]) 

        ############### Backward Pass ####################
        # update generator weights
        model.module.optimizer_G.zero_grad()
        loss_G.backward()
        model.module.optimizer_G.step()

        # update discriminator weights
        if not opt.no_gan:
            model.module.optimizer_D.zero_grad()
            loss_D.backward()
            model.module.optimizer_D.step()

        #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

        ############## Display results and errors ##########
        ### print out errors
        # if total_steps % opt.print_freq == 0:
        #     errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
        #     t = (time.time() - iter_start_time) / opt.batchSize
        #     visualizer.print_current_errors(epoch, epoch_iter, errors, t)
        #     visualizer.plot_current_errors(errors, total_steps)

        ### display output images
        if save_fake:
            # SH: changed the class to return the variables to visualize 
            visuals = model.module.get_current_visuals()
            visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        ### save loss tensorboard
        if total_steps % opt.save_latest_freq == 0:
            writer.add_scalars('loss_D',{'train':np.mean(loss_D_li[-100:])}, total_steps)
            writer.add_scalars('loss_D_fake',{'train':np.mean(loss_D_fake_li[-100:])}, total_steps)
            writer.add_scalars('loss_D_real',{'train':np.mean(loss_D_real_li[-100:])}, total_steps)
            writer.add_scalars('loss_G',{'train':np.mean(loss_G_li[-100:])}, total_steps)
            writer.add_scalars('loss_G_GAN',{'train':np.mean(loss_G_GAN_li[-100:])}, total_steps)
            writer.add_scalars('loss_G_VGG',{'train':np.mean(loss_G_VGG_li[-100:])}, total_steps)
            writer.add_scalars('loss_G_GAN_feat',{'train':np.mean(loss_G_GAN_feat_li[-100:])}, total_steps)
            writer.add_scalars('loss_G_imp',{'train':np.mean(loss_G_imp_li[-100:])}, total_steps)

            loss_D_li = []
            loss_D_fake_li = []
            loss_D_real_li = []
            loss_G_li = []
            loss_G_GAN_li = []
            loss_G_GAN_feat_li = []
            loss_G_VGG_li = []
            loss_G_imp_li = []
        
    # debug用 model save
    # out_path = os.path.join(save_dir, '_epoch' + str(epoch) + '_step' + str(total_steps) + '.pt')
    # torch.save(model.state_dict(), out_path)
    # exit()

    ### print out errors
    errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
    t = (time.time() - iter_start_time) / opt.batchSize
    visualizer.print_current_errors(epoch, epoch_iter, errors, t)
    visualizer.plot_current_errors(errors, total_steps)

    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save('epoch{}'.format(epoch))
        # prev_epoch = epoch - opt.save_epoch_freq * opt.num_checkpoint
        # if prev_epoch >= 0:
        #     model.module.delete_model(prev_epoch)
        # np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

        # out_path = os.path.join(save_dir, 'epoch' + str(epoch) + '_step' + str(total_steps) + '.pt')
        # torch.save(model.state_dict(), out_path)

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()
process_time = time.time() - process_start_time
print('all process -> {}:{}:{}'.format(int(process_time//3600), int((process_time%3600)//60), int((process_time%3600)%60)))
