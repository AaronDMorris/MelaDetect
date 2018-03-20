import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

tqdm.monitor_interval = 0

IMAGE_SIZE = 122

TRAIN_DIR = 'E:\\MelaDetect\\alldata_TRAINING_DATA'
TEST_DIR = 'E:\\MelaDetect\\alldata_TEST_DATA'

def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'Malignant' : return [1,0]
    elif word_label == 'Benign' : return [0,1]


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (IMAGE_SIZE, IMAGE_SIZE))
        training_data.append([np.array(img), np.array(label)])
        
    shuffle(training_data)

    np.save('alldata122COLOR_train_data.npy' , training_data)
    return training_data   
    
'''
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (IMAGE_SIZE, IMAGE_SIZE))
        #img = cv2.fastNlMeansDenoising(img, 5, 7)

        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
 
        training_data.append([np.array(img), np.array(label)])
        
    shuffle(training_data)
	
    np.save('alldata122COLOR_train_data.npy' , training_data)
    return training_data   
'''   
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        
        img_num = img.split('.')[0]
        
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (IMAGE_SIZE, IMAGE_SIZE))
        #img = cv2.fastNlMeansDenoising(img, 5, 7)
        testing_data.append([np.array(img), img_num])
    #shuffle(testing_data)              
    np.save('alldata244_test_data.npy', testing_data)
    return testing_data
  
train_data = create_train_data()

#test_data = process_test_data()