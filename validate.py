import glob
import os
import random
import cv2
import torch
import math
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
import torch
from numpy import loadtxt
import codecs
import math
from torch.autograd import Variable
import time
from unet_model import UNet
import matplotlib.pyplot as plt
#from torchviz import make_dot
import SSIM_ke
import pandas as pd
#from skimage.metrics import structural_similarity as ssim

def ssim(img1, img2):
    L = np.max(img1)
    C1 = (0.001*L)
    C2 = (0.03*L)**2
    ks = 5
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    kernel = np.ones((ks, ks), np.float64) / ks**2
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[ks:-ks, ks:-ks]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[ks:-ks, ks:-ks]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[ks:-ks, ks:-ks] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[ks:-ks, ks:-ks] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[ks:-ks, ks:-ks] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    plt.imshow(img1)
    #plt.show()
    plt.imshow(img2)
    #plt.show()
    plt.imshow(ssim_map)
    #plt.show()
    return ssim_map.mean()


def compare_L2(path1, path2, ssim_):
    cr = nn.MSELoss(reduction='mean')
    image1 = np.fromfile(path1, dtype=np.int32)
    if np.size(image1) > 144 * 144:
        image1 = np.reshape(image1, (144, 144, 3))[:, :, 0]
    else:
        image1 = np.reshape(image1, (144, 144))
    image2 = np.fromfile(path2, dtype=np.int32)
    if np.size(image2) > 144 * 144:
        image2 = np.reshape(image2, (144, 144, 3))[:, :, 0]
    else:
        image2 = np.reshape(image2, (144, 144))
    #plt.imshow(image1)
    #plt.show()
    #plt.imshow(image2)
    #plt.show()
    l2loss = np.sum((image1[40:104, 40:104] - image2[40:104, 40:104])**2)
    l2loss = l2loss/image1[40:104, 40:104].size
    if ssim_:
        l2loss = ssim(image1[40:104, 40:104], image2[40:104, 40:104])
        return l2loss
    image1 = torch.from_numpy(image1).float()
    image2 = torch.from_numpy(image2).float()
    #l2loss = cr(image1,image2)
    return l2loss


def compare_L2_emb(path1, path2):
    image1 = np.fromfile(path1, dtype=np.int32)
    if np.size(image1) > 144 * 144:
        image1 = np.reshape(image1, (144, 144, 3))[:, :, 0]
    else:
        image1 = np.reshape(image1, (144, 144))
    image2 = np.fromfile(path2, dtype=np.int32)
    if np.size(image2) > 144 * 144:
        image2 = np.reshape(image2, (144, 144,3))[:, :,0]
    else:
        image2 = np.reshape(image2, (144, 144))
    n,l2loss = 0,0

    #plt.imshow(image1)
    #plt.show()
    #plt.imshow(image2)
    #plt.show()

    for i in range(144):
        for j in range(144):
            if image2[i,j] > 1 and math.sqrt((i-72)**2+(j-72)**2) < 87.*0.9/2.:
                one = float(image2[i,j])
                two = float(image1[i,j])
                l2loss += (one-two)**2
                n+=1
    l2loss = l2loss/(n+0.10001)
    return l2loss




def compare_L2_(path1, path2):
    sum = 0
    ssim_ = 0
    for el1, el2 in zip(path1,path2):
        sum += compare_L2(el1, el2, ssim_=False) #compare_L2_emb(el1, el2)
        ssim_ += compare_L2(el1, el2, ssim_=True)
        #print(compare_L2(el1, el2, ssim_=True))
    return (sum/len(path1),ssim_/len(path1))


def calcul_norm(Nview, Size_ob):
    dict = [{'LSTM':False, 'unet_channels':1, 'LSTM_num_kernels':0, 'LSTM_num_layers':0, 'Rotate':False},
            {'LSTM': False, 'unet_channels': Nview, 'LSTM_num_kernels': 0, 'LSTM_num_layers': 0, 'Rotate':False}]
            #{'LSTM': True, 'unet_channels': 1, 'LSTM_num_kernels': 1, 'LSTM_num_layers': 1, 'Rotate':False},
            #{'LSTM': True, 'unet_channels': 5, 'LSTM_num_kernels': 5, 'LSTM_num_layers': 1, 'Rotate':False},
            #{'LSTM': True, 'unet_channels': 5, 'LSTM_num_kernels': 5, 'LSTM_num_layers': 5, 'Rotate':False},
            #{'LSTM': True, 'unet_channels': 1, 'LSTM_num_kernels': 5, 'LSTM_num_layers': 5, 'Rotate':False}]
    #dict = [{'LSTM': False, 'unet_channels': 1, 'LSTM_num_kernels': 0, 'LSTM_num_layers': 0, 'Rotate': False}]
    res = []
    for el in dict:
        name_for_save = f'data{Size_ob}/validation/unet/*_0{Nview}_*'+str(el).replace(" ", "") + '*'
        list_image = glob.glob(name_for_save)
        pathlabel = glob.glob(f"data{Size_ob}/validation/label/*.png")
        nomers = [btv.split('/')[-1].split('.png')[0] for btv in pathlabel]
        list_image = [glob.glob(f'data{Size_ob}/validation/unet/'+btv+f'_0{Nview}_*'+str(el).replace(" ", "") + '*')[0] for btv in nomers]
        res.append(compare_L2_(list_image, pathlabel))

    print(f'==== Nview = {Nview}, Size_ob = {Size_ob} =============')
    for el in res:
        print(el[0])
    for el in res:
        print(el[1])
    print('==============')


def draw_loss_function(file = 'Model.csv'):
    db = pd.read_csv(file,header=0,delimiter=',')
    print(db)
    print(len(db))
    db.plot(y=["loss", "val_loss",'val_Lung_variance','Lung_variance'], kind="line", figsize=(10, 10))
    #

    plt.show()

if __name__ == '__main__':
    #main(Nview)
    draw_loss_function(file='Model.csv')
