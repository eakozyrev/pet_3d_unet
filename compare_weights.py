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

def Lung_variance(y_true, y_pred):
    #y_true = y_true[1,150:160,40:50,40:50]
    var_true = K.var(y_true)
    #y_pred = y_pred[1,150:160,40:50,40:50]
    var_pred = K.var(y_pred)
    return var_pred / (var_true + K.epsilon())

model0 = keras.models.load_model("models_promts_atten_norm_scatt_random_emb/Model_0.h5")
print(model0.layers)


model1 = keras.models.load_model("models_promts_atten_norm_scatt_random_emb/Model_64.h5")

for i in range(0,len(model0.layers)):
    try:
        conv0 = model0.layers[i].get_weights()[0]
        conv1 = model1.layers[i].get_weights()[0]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle(f'{i}')
        p1 = np.sum(conv0[0][1],axis=0)
        p2 = np.sum(conv1[0][1],axis=0)
        ax1.imshow(p1)
        ax2.imshow(p2)
        ax3.imshow((p1 - p2)/(p1 + 0.001))
        plt.show()
    except: pass
