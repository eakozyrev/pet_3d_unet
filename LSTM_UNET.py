from keras import Model
from tensorflow.keras.layers import Conv3D, Input, MaxPooling3D, Dropout, concatenate, UpSampling3D
import tensorflow as tf

def Resnet(inputs,Nch):

    con1 = Conv3D(Nch, 3, padding='same', data_format="channels_first")(inputs)
    con2 = Conv3D(Nch, 3, padding='same', data_format="channels_first")(con1)
    return tf.keras.layers.PReLU(shared_axes=[2,3,4])(con2 + con1)


def Unet3D(inputs):

    Nch = inputs.shape[1]
    conv1 = Conv3D(16, 1, activation = 'ReLU', padding = 'same',data_format="channels_first", kernel_initializer=tf.keras.initializers.RandomNormal(mean=1./Nch, stddev=0.02, seed=123))(inputs)
    conv1 = Conv3D(16, 3, activation = 'ReLU', padding = 'same',data_format="channels_first")(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2),data_format='channels_first')(conv1)
    conv2 = Resnet(pool1,24)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2),data_format='channels_first')(conv2)
    conv3 = Resnet(pool2,64)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2),data_format='channels_first')(conv3)

    conv4 = Resnet(pool3,128)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2),data_format='channels_first')(drop4)

    conv5 = Resnet(pool4,512)
    drop5 = Dropout(0.5)(conv5)

    upsam1 = UpSampling3D(size = (2,2,2),data_format='channels_first')(drop5)
    up6 = Conv3D(128, 2, activation = 'ReLU', padding = 'same',data_format='channels_first')(upsam1)
    merge6 = concatenate([drop4,up6],axis=1)
    conv6 = Resnet(merge6,128)

    upsam2 = UpSampling3D(size = (2,2,2),data_format='channels_first')(conv6)
    up7 = Conv3D(64, 2, activation = 'ReLU', padding = 'same',data_format='channels_first')(upsam2)
    merge7 = concatenate([conv3,up7],axis=1)
    conv7 = Resnet(merge7,64)

    upsam3 = UpSampling3D(size = (2,2,2),data_format='channels_first')(conv7)
    up8 = Conv3D(32, 2, activation = 'ReLU', padding = 'same',data_format='channels_first')(upsam3)
    merge8 = concatenate([conv2,up8],axis=1)
    conv8 = Resnet(merge8,32)

    upsam4 = UpSampling3D(size = (2,2,2),data_format='channels_first')(conv8)
    up9 = Conv3D(16, 2, activation = 'ReLU', padding = 'same',data_format='channels_first')(upsam4)
    merge9 = concatenate([conv1,up9],axis=1)
    conv9 = Resnet(merge9,16)
    conv10 = Conv3D(1, 1, padding = 'same',data_format='channels_first',use_bias=False)(conv9)

    model = Model(inputs=inputs, outputs = conv10)

    return model


# Build model.
#x = tf.convert_to_tensor(np.ones((1,384,192,192, 1)))
#inputs = tf.keras.Input(shape=(384,192,192, 1), name='CT')
#model = Unet3D(inputs,num_classes=1)
#model.summary()
#print(model.predict(x))
# N layers = 1     Total params: 352,779
# N layers = 20    Total params: 352,798