import tensorflow as tf
import math
import numpy as np
import os
my_seed = 1324
os.environ['PYTHONHASHSEED'] = str(my_seed)
import random
random.seed(my_seed)
np.random.seed(my_seed)
tf.random.set_seed(my_seed)

images_for_train = '/spoolA/anon_explorer/dataset/promts_atten_norm_scatt_random_emb/'
ground_truth_for_train= '/spoolA/anon_explorer/dataset/promts_atten_norm_scatt_random_emb/ramla/'

image_for_validation = '/spoolA/anon_explorer/dataset/promts_atten_norm_scatt_random/'
ground_truth_for_validation = '/spoolA/anon_explorer/dataset/ramla/'

N_layers = 20
INPUT_PATCH_SIZE=(N_layers,512,144,144)
TRAINING_INITIAL_EPOCH=0
TRAING_EPOCH_in_batch=50
TRAING_EPOCH=100
TRAIN_CLASSIFY_LEARNING_RATE =1e-4

###----Resume-Training
RESUME_TRAINING = 0
RESUME_TRAIING_MODEL='models_promts_atten_norm_scatt_random_emb/Model_37.h5'


#TRAIN_CLASSIFY_LOSS=tf.keras.losses.binary_crossentropy()
OPTIMIZER=tf.keras.optimizers.Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE,epsilon=1e-7)

TRAINING_CSV='Model.csv'

