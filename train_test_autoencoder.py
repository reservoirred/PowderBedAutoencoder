import os
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import struct
import cv2
from numpy import expand_dims
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.patches import Rectangle
import random
import pandas
from scipy.signal import convolve2d
from sklearn.metrics import accuracy_score        
import sklearn
import sklearn.metrics
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2DTranspose,Subtract,concatenate,MaxPooling2D,RepeatVector,Add,Dropout,\
Dense,Flatten,Softmax,LeakyReLU,Input, Concatenate,AveragePooling2D,Lambda, Conv2D, \
BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D,GlobalAveragePooling2D,Add,\
RepeatVector,UpSampling3D,Lambda,Permute,SpatialDropout2D,MultiHeadAttention
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import regularizers
from tensorflow.keras.activations import sigmoid,softmax,relu,gelu
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping


print('success')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


main_folder = 'D:\\cmu07_buildfailure\\'
data_folder = main_folder+'powderbedimages\\'
ml_rows,ml_cols,chans = 64,64,1
batch_size = 10

image_folder = data_folder
valid_per = .3

sample_list = [x for x in os.listdir(image_folder) if x.endswith('.jpg')]

build_list = sample_list.copy()

sample_list = sample_list[1:1500]

indexes_valid = random.sample(range(0,len(sample_list)),int(valid_per*float(len(sample_list))))
indexes_train = [x for x in range(0,len(sample_list)) if x not in indexes_valid]
training_list = [x for ind,x in enumerate(sample_list) if ind in indexes_train]
valid_list = [x for ind,x in enumerate(sample_list) if ind in indexes_valid]

print(len(sample_list))

tform_matrix = np.loadtxt(main_folder+'tform_bed.txt')
# actual tform values below
#[1.35786458946702	-0.0184261429440753	-1.20688534505464e-05
#0.114196836230360	1.54483936478397	0.000201776048811541
#-268.623676745108	-149.886118613164	1]

def get_sample(sample_name,batch_size,aug):
    img = np.array(cv2.imread(data_folder+sample_name,-1),dtype='float32')/255
    img = cv2.warpPerspective(img, tform_matrix.transpose(), img.shape)
    img = img[75:1000,75:1000]
    if aug:
       k = random.sample(range(0,360,90),1)[0]
       M = cv2.getRotationMatrix2D((img.shape[0]/2,img.shape[1]/2), k, 1)
       img = cv2.warpAffine(img, M, (img.shape[0],img.shape[1]))  
     
    ix = np.random.randint(0,img.shape[0]-ml_rows,batch_size)
    iy = np.random.randint(0,img.shape[1]-ml_rows,batch_size)
    
    img_list = [img[x[0]:x[0]+ml_rows,x[1]:x[1]+ml_rows] for x in zip(ix,iy)]
    img_stack = np.asarray(img_list)
    return img_stack

test = get_sample(training_list[0],10,True)

def batch_generator(sample_list,batch_size,aug):
    while True:
        random.shuffle(sample_list)
        batch_filenames = np.random.choice(sample_list,1)
        batch_input = np.expand_dims(get_sample(batch_filenames[0],batch_size,aug),axis=-1)
        # for i_sample in batch_filenames:
        #     i_array = get_sample(i_sample,batch_size,aug)
        #     batch_input+=[i_array]
        # batch_input = np.array(batch_input,dtype='float32')
        yield (batch_input,batch_input)

test1,test2 = next(batch_generator(training_list,10,False))


#%% network modeling
regterm = 0
def conv3x3(input_layer,numel_filters):
    x = Conv2D(numel_filters,(3,3),strides=1,padding='same',
               kernel_regularizer=regularizers.l2(regterm),bias_regularizer=regularizers.l2(regterm))(input_layer)
    x = BatchNormalization(axis=-1,momentum = .5)(x) 
    x = LeakyReLU(alpha=.3)(x)
    return x

def conv3x3_down(input_layer,numel_filters):
    x = Conv2D(numel_filters,(3,3),strides=1,padding='same',
               kernel_regularizer=regularizers.l2(regterm),bias_regularizer=regularizers.l2(regterm))(input_layer)
    x = BatchNormalization(axis=-1,momentum = .5)(x)
    x = LeakyReLU(alpha=.3)(x)
    x = AveragePooling2D(pool_size=(2,2),padding='valid')(x)
    return x

def conv3x3_skipconnect(input_layer,numel_filters):
    x = UpSampling2D(size=(2, 2))(input_layer)
    x = Conv2D(numel_filters,(3,3),strides=1,padding='same',
               kernel_regularizer=regularizers.l2(regterm),bias_regularizer=regularizers.l2(regterm))(x)
    x = BatchNormalization(axis=-1,momentum = .5)(x)
    x = LeakyReLU(alpha=.3)(x)
    return x

def regressionLayer(input_layer):
    x = Conv2D(1,(1,1),strides=1,padding='same',activation='sigmoid',
              kernel_regularizer=regularizers.l2(regterm),bias_regularizer=regularizers.l2(regterm))(input_layer)
    return x

f1,f2,f3,f4,f5,f6,f7 = 8,16,32,64,128,128,128
layer_input = Input((ml_rows,ml_rows,1))
pre_process = conv3x3(layer_input,f1) #256
down_1 = conv3x3_down(pre_process,f2) #128
down_2 = conv3x3_down(down_1,f3) #64
down_3 = conv3x3_down(down_2,f4) #32
down_4 = conv3x3_down(down_3,f5) #16

middle = conv3x3(down_4,f7)

up_4 = conv3x3_skipconnect(middle,f4) #32
up_3 = conv3x3_skipconnect(up_4,f3) #64
up_2 = conv3x3_skipconnect(up_3,f2) #128
up_1 = conv3x3_skipconnect(up_2,f1) #256

post_process = conv3x3(up_1,f1)
classification = regressionLayer(post_process)

network = Model(layer_input,classification)
network.summary()

from tensorflow.keras.utils import plot_model
folder = 'D:\\cmu07_buildfailure\\'
plot_model(network, to_file=folder+'model_plot.png', show_shapes=True, show_layer_names=True,show_layer_activations=False)



adam = Adam(lr=0.002, beta_1=0.5)
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.metric = []

    # def on_train_end(self, logs={}):
    # def on_epoch_begin(self, logs={}):
    # def on_epoch_end(self, logs={}):
    # def on_batch_begin(self, logs={}):
    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.loss.append(logs.get('mae'))
        self.loss.append(logs.get('mse'))

network.compile(loss='mae', optimizer=adam, metrics=['mae', 'mse'])
history_call = LossHistory()
val_stop_call = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto', restore_best_weights=True)
reduce_lr_call = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=.000002)
traininghistory = network.fit_generator(batch_generator(training_list, 25,True), epochs=100,
                                    steps_per_epoch=1000,
                                    validation_data=batch_generator(valid_list, 25,True),
                                    validation_steps=250,
                                    callbacks=[history_call, val_stop_call, reduce_lr_call])



#%% test on full field image
def network_on_fov(img,network):
    
    ix = np.arange(0,img.shape[0]-ml_rows,ml_rows)
    iy = np.arange(0,img.shape[0]-ml_rows,ml_rows)
    
    mse_map = np.zeros(img.shape)
    
    xx,yy = np.meshgrid(ix,iy)
    img = np.expand_dims(img,axis=0)
    img = np.expand_dims(img,axis=-1)
    img_list = [np.abs(network.predict(img[0:1,x[0]:x[0]+ml_rows,x[1]:x[1]+ml_rows,0:1])-img[0:1,x[0]:x[0]+ml_rows,x[1]:x[1]+ml_rows,0:1]) for x in zip(xx.flatten(),yy.flatten())]    
    
    for x in zip(xx.flatten(),yy.flatten(),img_list):
        mse_map[x[0]:x[0]+ml_rows,x[1]:x[1]+ml_rows] = x[2][:,:,0]
    
    return mse_map


img = np.array(cv2.imread(data_folder+'SI186620211116123131_00118_20211116T131110.575000_Cam0_Powderbed RecoatingEnd.jpg',-1),dtype='float32')/255
img = cv2.warpPerspective(img, tform_matrix.transpose(), img.shape)
img = img[75:1000,75:1000]


img_pred = network_on_fov(img,network)
plt.imshow(img_pred,vmin=0,vmax=1)
plt.colorbar()

img = np.array(cv2.imread(data_folder+'SI186620211116123131_01694_20211116T210742.523000_Cam0_Powderbed RecoatingEnd.jpg',-1),dtype='float32')/255
img = cv2.warpPerspective(img, tform_matrix.transpose(), img.shape)
img = img[75:1000,75:1000]

img_pred = network_on_fov(img,network)
plt.imshow(img_pred,vmin=0,vmax=1)


fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
pcm = axs[0].imshow(img,vmin=0,vmax=1)
fig.colorbar(pcm, ax=axs[0])
pcm = axs[1].imshow(img_pred,vmin=0)
fig.colorbar(pcm, ax=axs[1])


max_error = []

for x in build_list[:]:
    print(x)
    img = np.array(cv2.imread(data_folder+x,-1),dtype='float32')/255
    img = cv2.warpPerspective(img, tform_matrix.transpose(), img.shape)
    img = img[75:1000,75:1000]
    img_pred = network_on_fov(img,network)
    max_error.append(np.max(img_pred))





plt.plot(max_error)



