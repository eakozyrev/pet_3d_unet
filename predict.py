from LSTM_UNET import Unet3D
from config import*
import os
import datetime
import tensorflow as tf
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib
from astropy.visualization import simple_norm
from pandas.core.common import flatten
from train import *
from config import *

dataset = create_dataset(['/data/MyBook/Data/anon_explorer/dataset/ramla/',
                        '/data/MyBook/Data/anon_explorer/dataset/promts_norm_random/'],
                         True,1,N_layers)

model = keras.models.load_model("models_promts_atten_norm_scatt_random_emb_20channels/Model_50.h5", compile=False)

def plot_slices(arr):
    fig, axs = plt.subplots(3, 4)
    ii = 86 + 50
    for i in range(3):
        for j in range(4):
            btv = arr[0, 0, ii, 25:110]
            norm = simple_norm(btv, 'sqrt')
            im = axs[i, j].imshow(btv, norm=norm)
            ii += 25

def Lung_variance(y_true, y_pred):
    y_true = y_true[0,0,150:160,70:80,70:80]
    print(y_true.shape)
    print(y_true)
    var_true = K.var(y_true)
    y_pred = y_pred[0,0,150:160,70:80,70:80]
    var_pred = K.var(y_pred)
    return var_pred / (var_true + K.epsilon())



label_ = dataset[0]
arr_ = dataset[1]

loss_fn = tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size')
model.compile(optimizer=OPTIMIZER, loss=loss_fn, metrics=['mean_squared_error',Lung_variance])

plot_slices(label_)
plot_slices(np.sum(arr_,axis=(0,1),keepdims=True))
img = model.predict(arr_)
print('max(img) = ',img.max())
plot_slices(img)

plt.show()


dot_img_file = 'Model_1.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
fw = tf.summary.create_file_writer(f'logs/fit/{filename}')
N_layers = len(model.layers)
print('N_layers = ',N_layers)
with fw.as_default():
    for i in range(N_layers):
        try:
            t = model.layers[i].get_weights()[0]
            print(i, t.shape)
            #if i == 1:
                #fig, axs = plt.subplots(1,1)
                #plt.plot(model.layers[i].get_weights()[0][0,0,0,:,0])
                #plt.show()
        except: pass

#for i in range(N_layers-20,N_layers-10):
#    model.layers[i].set_weights(np.multiply(model.layers[i].get_weights(),3*math.sin(i)))

