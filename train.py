from __future__ import absolute_import, division, print_function, unicode_literals
from config import*
import os
import datetime
from LSTM_UNET import Unet3D
import numpy as np
import random
import glob
import gc
import torch
import tensorflow as tf
from keras import backend as K


def Lung_variance(y_true, y_pred):
    y_true = y_true[0,0,150:160,70:80,70:80]
    print(y_true.shape)
    print(y_true)
    var_true = K.var(y_true)
    y_pred = y_pred[0,0,150:160,70:80,70:80]
    var_pred = K.var(y_pred)
    return var_pred / (var_true + K.epsilon())


def getting_list(path):
    a=[file for file in os.listdir(path) if file.endswith('.i')]
    all_tfrecoeds=random.sample(a, len(a))
    #all_tfrecoeds.sort(key=lambda f: int(filter(str.isdigit, f)))
    list_of_tfrecords=[]
    for i in range(len(all_tfrecoeds)):
        tf_path=path+all_tfrecoeds[i]
        list_of_tfrecords.append(tf_path)
    return list_of_tfrecords


def read_sample(listfiles, number, N_layers):

    label_ = glob.glob(listfiles[0] + f'*_{number}.i')
    print('label_ = ',label_)
    label = np.fromfile(label_[0], dtype=np.float32)
    label = np.reshape(label, (1, 1, 356, 150, 150))
    label = np.pad(label, [(0, 0), (0, 0), (78, 78), (0, 0), (0, 0)])
    label = np.reshape(label, (1, 1, 512, 150, 150))[:, :, :, 3:-3, 3:-3]
    npmean = np.mean(label)
    label = label/npmean

    arr_ = glob.glob(listfiles[1] + f'*_{number}.i')
    print('image = ',arr_)
    arr = np.fromfile(arr_[0], dtype=np.float32)
    arr = np.reshape(arr, (9, 20, 356, 150, 150))
    arr = np.pad(arr, [(0, 0), (0, 0), (78, 78), (0, 0), (0, 0)])
    if N_layers==20:
        arr = np.sum(arr[1:-1],axis=0,keepdims=True)
    else:
        arr = np.sum(arr[1:-1], axis=(0,1), keepdims=True)
    arr = np.reshape(arr, (1, N_layers, 512, 150, 150))[:, :, :, 3:-3, 3:-3]
    npmean = np.mean(arr)
    arr = arr/npmean

    if len(listfiles) > 2:
        ct_ = glob.glob(listfiles[2] + f'*{number}.i')
        print('ct_ =', ct_)
        ct = np.fromfile(ct_[0], dtype=np.float32)
        ct = np.reshape(ct, (1, 1, 356, 150, 150))
        ct = np.pad(ct, [(0, 0), (0, 0), (78, 78), (0, 0), (0, 0)])
        ct = np.reshape(ct, (1, 1, 512, 150, 150))[:, :, :, 3:-3, 3:-3]
        arr = np.concatenate(arr,ct,axis=0)

    return (label,arr)



def create_dataset(listfiles, is_validation, number, Nlayers):
    list_label = sorted(glob.glob(listfiles[0] + '/*.i'))
    #print('list_label = ', list_label)
    Nimages = len(list_label)
    for el in listfiles:
        if Nimages > len(sorted(glob.glob(el + '/*.i'))):
            print('======================== ERROR =================')
            print('len(list_label) < N labels')
            Nimages = len(list_label)

    if is_validation:
        valid_dataset = read_sample(listfiles, 1, Nlayers)
        return valid_dataset

    train_dataset_l = []
    train_dataset_i = []
    for el in np.random.choice(range(2,Nimages+1),number):
        tmp = read_sample(listfiles, el, N_layers)
        train_dataset_l.append(tmp[0])
        train_dataset_i.append(tmp[1])

    train_dataset_l = np.concatenate(train_dataset_l,axis=0)
    train_dataset_i = np.concatenate(train_dataset_i,axis=0)
    train_dataset = [train_dataset_l,train_dataset_i]

    return train_dataset



def Training(N_layers):

    path_to_save_models = "models_" + images_for_train.split('/')[-2] + f'_{N_layers}channels'
    os.system('mkdir '+ path_to_save_models)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.6

    csv_logger = tf.keras.callbacks.CSVLogger(TRAINING_CSV, append=True)

    valdataset = create_dataset([ground_truth_for_validation,image_for_validation],
                                    is_validation=True,number=2,Nlayers=N_layers)

    loss_fn = tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size')
    log_dir = "log/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    inputs = tf.keras.Input(shape=INPUT_PATCH_SIZE, name='T')
    Model_3D = Unet3D(inputs)
    Model_3D.compile(optimizer=OPTIMIZER, loss=loss_fn, metrics=['mean_squared_error',Lung_variance])
    print(Model_3D.summary())
    dot_img_file = 'model_1.png'
    tf.keras.utils.plot_model(Model_3D, to_file=dot_img_file, show_shapes=True)

    filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fw = tf.summary.create_file_writer(f'logs/fit/{filename}')

    if RESUME_TRAINING==1:
        Model_3D.load_weights(RESUME_TRAIING_MODEL)
        N_layers = len(Model_3D.layers)
        # for i in range(N_layers-12):
        #b Model_3D.layers[i].trainable = False
        Model_3D.compile(optimizer=OPTIMIZER, loss=loss_fn, metrics=['mean_squared_error',Lung_variance])

    for _ in range(TRAINING_INITIAL_EPOCH,TRAING_EPOCH):
        np.random.seed(_)
        print('-------------------- epoch = ',_)
        dataset = create_dataset([ground_truth_for_train, images_for_train],
                                 is_validation=False, number=1, Nlayers=N_layers)
        history = Model_3D.fit(dataset[1],dataset[0],
                   verbose=1,
                   epochs=TRAING_EPOCH_in_batch,
                   validation_steps=1,
                   callbacks=[csv_logger, tensorboard_callback],
                   validation_data=(valdataset[1],valdataset[0]))

        Model_3D.save(f'{path_to_save_models}/Model_{_}.h5')
        print(tf.config.experimental.get_memory_info("GPU:0"))
        del dataset
        gc.collect()
        torch.cuda.empty_cache()
        tf.compat.v1.reset_default_graph()
        print(tf.config.experimental.get_memory_info("GPU:0"))





if __name__ == '__main__':
   Training(N_layers)
