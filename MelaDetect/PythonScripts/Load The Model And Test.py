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


#Load the images, that will make up the testing dataset
#test_data = np.load('COLORalldata_melanet_2layer.model')

malignant_test_data = np.load('COLORmalignant_test_data.npy')
benign_test_data = np.load('COlORbenign_test_data.npy')

tf.reset_default_graph()

#MelaNet Deep Convolutional Neural Network architecture begins
#with tf.device('/gpu:0'):  

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



#model.load('2-melanet_COLORalldata_2layerFC.model')
model.load('with_y_pred_2-melanet_COLORalldata_2layerFC.model')


malignant_test2_x = np.array([i[0] for i in malignant_test_data]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
malignant_test2_y = [i[1] for i in malignant_test_data]

benign_test2_x = np.array([i[0] for i in benign_test_data]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
benign_test2_y = [i[1] for i in benign_test_data]

'''

for i in test2_x:
    print(model.predict(np.reshape(i, (-1, IMAGE_SIZE, IMAGE_SIZE, 1))))
    #if np.take(currentPrediction, 0) > 0.5:
    #print('Not Cancer\nWith %f accuracy', np.take(currentPrediction, 0))
    #cat += 1
    #else: 
    #print('Cancer\nWith ', np.take(currentPrediction, 1) , ' accuracy')
    #dog += 1
    
#print('The toal number of cancer cases predicted: ', cat)
#print('The toal number of non-cancer cases predicted: ', dog)   
'''

#model.load('alldata_melanet_2layer.model')

#print(model.predict(np.reshape(test2_x[0], (-1, IMAGE_SIZE, IMAGE_SIZE, 1))))
#print(model.predict(np.reshape(test2_x[5], (-1, IMAGE_SIZE, IMAGE_SIZE, 1))))

'''
cancer = 0
non_cancer = 0

for i in test2_x:
    currentPrediction = model.predict(np.reshape(i, (-1, IMAGE_SIZE, IMAGE_SIZE, 1)))
    if np.take(currentPrediction, 0) > 0.26810565591:
        print('Cancer\nWith ', np.take(currentPrediction, 1) , ' accuracy')
        cancer += 1
        
    else: 
        print('Not Cancer\nWith %f accuracy', np.take(currentPrediction, 0))
        non_cancer += 1
    
print('The toal number of cancer cases predicted: ', cancer)
print('The toal number of non-cancer cases predicted: ', non_cancer)
''''''
cancer = 0
non_cancer = 0

for i in malignant_test2_x:
    currentPrediction = model.predict(np.reshape(i, (-1, IMAGE_SIZE, IMAGE_SIZE, 1)))
    if np.take(currentPrediction, 1) > 0.26810565591:
        print('Cancer\nWith ', np.take(currentPrediction, 1) , ' accuracy')
        cancer += 1
        
    else: 
        print('Not Cancer\nWith %f accuracy', np.take(currentPrediction, 0))
        non_cancer += 1
    
print('DETECTED RIGHT: ', cancer)
print('DETECTED WRONG: ', non_cancer)

cancer = 0
non_cancer = 0

for i in benign_test2_x:
    currentPrediction = model.predict(np.reshape(i, (-1, IMAGE_SIZE, IMAGE_SIZE, 1)))
    if np.take(currentPrediction, 0) > 0.26810565591:
        print('Wrong: ', np.take(currentPrediction, 1) , ' ')
        cancer += 1
        
    else: 
        print('Correct:', np.take(currentPrediction, 0))
        non_cancer += 1
    
print('DETECTED WRONG: ', cancer)
print('DETECTED RIGHT: ', non_cancer)
'''
print(model.predict(np.reshape(benign_test2_x[1875], (-1, IMAGE_SIZE, IMAGE_SIZE, 3))))
print(benign_test2_y[1875])
'''
print('Malignant:')
imageNumber = 0
malignantCount = 0
benignCount = 0

for i in malignant_test2_x:
    currentPrediction = model.predict(np.reshape(i, (-1, IMAGE_SIZE, IMAGE_SIZE, 3)))
    if(currentPrediction[0][0] > 0.183):
    #if(currentPrediction[0][0] > 0.5):
        imageNumber += 1
        print('Malignant' + str(currentPrediction) + ' ' + str(imageNumber))
        malignantCount += 1
    else:
        imageNumber += 1
        benignCount += 1
        print('Benign' + str(currentPrediction) + ' ' + str(imageNumber) + 'WRONG')
    #print(model.predict(np.reshape(i, (-1, IMAGE_SIZE, IMAGE_SIZE, 3))))

print('Benign:')
imageNumber = 0
for i in benign_test2_x:
    currentPrediction = model.predict(np.reshape(i, (-1, IMAGE_SIZE, IMAGE_SIZE, 3)))
    if(currentPrediction[0][0] > 0.183):
    #if(currentPrediction[0][0] > 0.5):
        imageNumber += 1
        print('Malignant' + str(currentPrediction) + ' ' + str(imageNumber) + 'WRONG')
        malignantCount += 1
    else:
        imageNumber += 1
        benignCount += 1
        print('Benign' + str(currentPrediction) + ' ' + str(imageNumber))
        
        
print('Benign:' + str(benignCount))  
print('Malignant:' + str(malignantCount))        
'''

'''
for i in test2_x:
    print(model.predict(np.reshape(i, (-1, IMAGE_SIZE, IMAGE_SIZE, 1))))
'''
'''
#for i in test2_y:
#    print(i)     
'''

                 

