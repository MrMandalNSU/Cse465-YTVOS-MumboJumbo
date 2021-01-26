import os
import sys
import numpy as np
import cv2
import json
import random
import torch
import torch.utils.data
import torchvision.models as models
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
import glob
from torchvision.utils import save_image
import configuration as cfg
from utils.dataloader5fps import dataLoader_5fps
from utils.initializer import Initializer, Encoder, Decoder
from utils.convlstm import *
from utils.ensemble import *

# added by Team-MumboJumbo
import router

batch_size = 4
epoch = cfg.epoch

transform_rgb = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop((256, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transforms_seg = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop((256, 448)),
    transforms.ToTensor()])

dset_train = dataLoader_5fps(cfg.JPEGtrain_5fps, cfg.Anntrain_5fps, cfg.json_path, transform_rgb, transforms_seg)

train_loader = DataLoader(dset_train,
                          batch_size=batch_size,
                          shuffle=True)

initializer = Initializer()
encoder = Encoder()
convlstm = ConvLSTMCell(input_size=512,
                        hidden_size=512)
decoder = Decoder()

model = MyEnsemble(initializer, encoder, convlstm, decoder)

if cfg.cuda_enable:
    model.cuda()

# print(model)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


def train(epoch):
    # optimizer =exp_lr_scheduler(optim.Adam(model.parameters()), epoch, init_lr=0.0001, lr_decay_epoch=30)
    model.train()
    train_loss = 0
    for batch_idx, (initialMask, initialRGB, segData, rgbData) in enumerate(train_loader):
        if cfg.cuda_enable:
            rgb = Variable(initialRGB).cuda()
            mask = Variable(initialMask).cuda()
            rgbData = Variable(rgbData).type(torch.FloatTensor).cuda()
            maskData = Variable(segData).type(torch.FloatTensor).cuda()
        else:
            rgb = Variable(initialRGB)
            mask = Variable(initialMask)
            rgbData = Variable(rgbData).type(torch.FloatTensor)
            maskData = Variable(segData).type(torch.FloatTensor)

        output = model(rgb, mask, rgbData)
        optimizer.zero_grad()

        loss = F.binary_cross_entropy(output, maskData)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        temp = torch.cat((output[:, 1, :, :, :], maskData[:, 1, :, :, :]), 0)

        # save_image(temp, '/home/shashank/shashank/AdvCV/Assign2/reconsImgs/recons_%d.png' % (epochs))

        # block added by Team-MumboJumbo

        recons_path = os.path.join(router.data_root, 'recons_{}.png'.format(epoch))
        save_image(temp, recons_path)

        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(initialMask), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(initialMask)))

    print('================> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

    if (epoch % 1 == 0):
        # torch.save(model.state_dict(), '/home/shashank/shashank/AdvCV/Assign2/weights/youtubeVOSModel_trial_3_%d.pth' %(epoch))

        # block added by Team-MumboJumbo

        model_path = os.path.join(router.model_root, 'youtubeVOSModel_trial_3_{}.pth'.format(epoch))
        torch.save(model.state_dict(), model_path)


for epochs in range(1, epoch + 1):
    train(epochs)
