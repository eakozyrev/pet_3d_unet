import numpy as np
from unet_model import UNet
from LSTM_UNET import LSTM_UNET
from dataset import Data_Loader
from torch import optim
import torch.nn as nn
import torch
from train import *
from gen_fig_2 import *
from shutil import copyfile
from predict import *
import os
from Conv_LSTM import *
from LSTM_UNET import *
from torchsummary import summary
from validate import calcul_norm

def initialize_weights(m):
    classname = m.__class__.__name__
    #print(classname)
    #if classname.find('UNet') == -1 and classname.find('ReLU') == -1 and classname.find('Sequential') == -1 and classname.find('DoubleConv') == -1 and classname.find('MaxPool2d') == -1 and classname.find('Down') == -1 and classname.find('Up') == -1 and classname.find('OutConv') == -1 and classname.find('ConvLSTMCell') == -1 and classname.find('ConvLSTM') == -1:
    try:
        m.weight.data.normal_(0, 0.0001)
        m.bias.data.normal_(0, 0.0001)
    except: pass
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if n == 0:
            return
        m.weight.data.normal_(0, math.sqrt(2. / n))

def train(Nview, Size_ob):
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    dict = [{'LSTM':False, 'unet_channels':1, 'LSTM_num_kernels':0, 'LSTM_num_layers':0, 'Rotate':False},
            {'LSTM': False, 'unet_channels': Nview, 'LSTM_num_kernels': 0, 'LSTM_num_layers': 0, 'Rotate':False}]
            #{'LSTM': True, 'unet_channels': 1, 'LSTM_num_kernels': 1, 'LSTM_num_layers': 1, 'Rotate':False},
            #{'LSTM': True, 'unet_channels': 5, 'LSTM_num_kernels': 5, 'LSTM_num_layers': 1, 'Rotate':False},
            #{'LSTM': True, 'unet_channels': 5, 'LSTM_num_kernels': 5, 'LSTM_num_layers': 5, 'Rotate':False},
            #{'LSTM': True, 'unet_channels': 1, 'LSTM_num_kernels': 5, 'LSTM_num_layers': 5, 'Rotate':False},


    #dict = [{'LSTM':False, 'unet_channels':1, 'LSTM_num_kernels':0, 'LSTM_num_layers':0, 'Rotate':False}]
    for el in dict:
        Net1 = LSTM_UNET(el).to(device=device)
        Net1.apply(initialize_weights)
        name_for_save = f'results{Size_ob}/{Nview}_' + str(el).replace(" ","")
        print('name_for_save = ',name_for_save)
        train_nn(Net1, device, f'data{Size_ob}/', epochs=30, batch_size=25, path = name_for_save,Nview = Nview)
        #summary(Net1,(4,144,144))


def predict_all(Nview, Size_ob):
    dict = [{'LSTM':False, 'unet_channels':1, 'LSTM_num_kernels':0, 'LSTM_num_layers':0, 'Rotate':False},
            {'LSTM': False, 'unet_channels': Nview, 'LSTM_num_kernels': 0, 'LSTM_num_layers': 0, 'Rotate':False}]
            #{'LSTM': True, 'unet_channels': 1, 'LSTM_num_kernels': 1, 'LSTM_num_layers': 1, 'Rotate':False},
            #{'LSTM': True, 'unet_channels': 5, 'LSTM_num_kernels': 5, 'LSTM_num_layers': 1, 'Rotate':False},
            #{'LSTM': True, 'unet_channels': 5, 'LSTM_num_kernels': 5, 'LSTM_num_layers': 5, 'Rotate':False},
            #{'LSTM': True, 'unet_channels': 1, 'LSTM_num_kernels': 5, 'LSTM_num_layers': 5, 'Rotate':False}]
    #dict = [{'LSTM':True, 'unet_channels':1, 'LSTM_num_kernels':0, 'LSTM_num_layers':0, 'Rotate':True}]
    for el in dict:
        name_for_save = f'results{Size_ob}/{Nview}_'+str(el).replace(" ", "")
        predict(f"data{Size_ob}/", name_for_save, el,Nview = Nview)

def draw_valid(Nview,Size_ob,num):
    dict = [{'LSTM':False, 'unet_channels':1, 'LSTM_num_kernels':0, 'LSTM_num_layers':0, 'Rotate':False},
            {'LSTM': False, 'unet_channels': Nview, 'LSTM_num_kernels': 0, 'LSTM_num_layers': 0, 'Rotate':False}]
            #{'LSTM': True, 'unet_channels': 1, 'LSTM_num_kernels': 1, 'LSTM_num_layers': 1, 'Rotate':False},
            #{'LSTM': True, 'unet_channels': 5, 'LSTM_num_kernels': 5, 'LSTM_num_layers': 1, 'Rotate':False},
            #{'LSTM': True, 'unet_channels': 5, 'LSTM_num_kernels': 5, 'LSTM_num_layers': 5, 'Rotate':False},
            #{'LSTM': True, 'unet_channels': 1, 'LSTM_num_kernels': 5, 'LSTM_num_layers': 5, 'Rotate':False}]
    arr = [f"data{Size_ob}/validation/label/{num}.png",f"data{Size_ob}/validation/image_sum/{num}.png"]
        #   f"data{Size_ob}/validation/label_emb/{num}.png"]
    for el in dict:
        name_for_save = f'data{Size_ob}/validation/unet/{num}_0' + str(Nview) + '_' + str(el).replace(" ", "") + '.png'
        arr.append(name_for_save)
    print(arr)
    draw_fig(arr)


def draw_loss(Nview,Size_ob):
    dict = [{'LSTM': False, 'unet_channels': 1, 'LSTM_num_kernels': 0, 'LSTM_num_layers': 0, 'Rotate': False},
            {'LSTM': False, 'unet_channels': Nview, 'LSTM_num_kernels': 0, 'LSTM_num_layers': 0, 'Rotate': False}]
            #{'LSTM': True, 'unet_channels': 1, 'LSTM_num_kernels': 1, 'LSTM_num_layers': 1, 'Rotate': False},
            #{'LSTM': True, 'unet_channels': 5, 'LSTM_num_kernels': 5, 'LSTM_num_layers': 1, 'Rotate': False},
            #{'LSTM': True, 'unet_channels': 5, 'LSTM_num_kernels': 5, 'LSTM_num_layers': 5, 'Rotate': False},
            #{'LSTM': True, 'unet_channels': 1, 'LSTM_num_kernels': 5, 'LSTM_num_layers': 5, 'Rotate': False}]
    for el in dict:
        name = f'results{Size_ob}/{Nview}_' + str(el).replace(" ", "")
        print(name+".dat")
        draw_loss_from_file(name+".dat")
    plt.show()

    #Net.load_rnn()
    #summary(Net, (1, 10, 144, 144))
    #

def draw_loss_valid(Size_ob):
    Nvw = [2,5,10,20]
    for el1 in Nvw:
        dict = [{'LSTM': False, 'unet_channels': 1, 'LSTM_num_kernels': 0, 'LSTM_num_layers': 0, 'Rotate': False},
                {'LSTM': False, 'unet_channels': el1, 'LSTM_num_kernels': 0, 'LSTM_num_layers': 0, 'Rotate': False}]
        for el in dict:
            name = f'results{Size_ob}/{el1}_' + str(el).replace(" ", "")+".dat"
            plot_valid(name,el['unet_channels'])
    plt.legend()
    plt.show()


if __name__ == "__main__":

    Nview = 20
    Size_ob = 2

    #draw_loss_valid(Size_ob)

    #train(Nview,Size_ob)
    #predict_all(Nview,Size_ob)
    draw_valid(Nview,Size_ob,9)
    #draw_loss(Nview,Size_ob)
    #calcul_norm(Nview,Size_ob)
    calcul_norm(2, Size_ob)
    calcul_norm(5, Size_ob)
    calcul_norm(10, Size_ob)
    calcul_norm(20, Size_ob)


    #train(2, 50)
    #train(5, 50)
    #train(10, 50)
    #train(20, 50)
    #train(2, 2)
    #train(5, 2)
    #train(10, 2)
    #train(20, 2)


