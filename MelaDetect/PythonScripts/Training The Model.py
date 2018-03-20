import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

tqdm.monitor_interval = 0
IMAGE_SIZE = 122
LEARNING_RATE = 1e-3
KEEP_RATE = 0.8
NUM_EPOCHS = 5


MODEL_NAME = 'MelignantOrBenign-{}-{}.model'.format(LEARNING_RATE, 'with_y_pred_alldata122COLOR-1203-run1-conv4L-maxpool-5epoch-dropout')

#Load the images, that have already been processed into a Numpy object
training_data = np.load('alldata122COLOR_train_data.npy')

#Ensure our graph has been reset
tf.reset_default_graph() 

#MelaNet Deep Convolutional Neural Network architecture begins
melanet = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name='input')

melanet = conv_2d(melanet, 32, 5, activation='relu')
melanet = max_pool_2d(melanet, 5)

melanet = conv_2d(melanet, 64, 5, activation='relu')
melanet = max_pool_2d(melanet, 5)

melanet = conv_2d(melanet, 128, 5, activation='relu')
melanet = max_pool_2d(melanet, 5)

melanet = conv_2d(melanet, 64, 5, activation='relu')
melanet = max_pool_2d(melanet, 2)

melanet = fully_connected(melanet, 1024, activation='relu')
melanet = dropout(melanet, KEEP_RATE)

melanet = fully_connected(melanet, 2, activation='softmax')
melanet = regression(melanet, optimizer='adam', learning_rate=LEARNING_RATE, loss='categorical_crossentropy', name='targets')


model = tflearn.DNN(melanet, tensorboard_dir='log')

#MelaNet Deep Convolutional Neural Network architecture ends


#Assign all of the images; except the last 1297(15%), for the training data
training = training_data[:-1297]
#Assign the remaining 15% of images for the validation data
validation = training_data[-1297:]

X = np.array([i[0] for i in training]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
Y = [i[1] for i in training]     

validation_x = np.array([i[0] for i in validation]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
validation_y = [i[1] for i in validation]


model.fit({'input': X}, {'targets': Y}, n_epoch=NUM_EPOCHS, validation_set=({'input': validation_x}, {'targets': validation_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]
model.save('with_y_pred_2-melanet_COLORalldata_2layerFC.model')


