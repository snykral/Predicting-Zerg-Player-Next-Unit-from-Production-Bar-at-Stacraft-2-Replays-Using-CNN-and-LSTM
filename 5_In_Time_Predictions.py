import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import models

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
from os import listdir


import pyautogui

import time


GAME_PATH = '/Game4/'



time.sleep(6)


DATA_PATH = 'Games_Img/'
#LISTED_DIR = list(set(os.listdir(DATA_PATH))-{'desktop.ini'})
LABELS_CATEGORY = listdir('Labeled_Img')

smallestW = 47
smallestH = 50

#IMAGES
def decode_img(img, size_flag):
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_jpeg(img, channels=3)
  # Resize the image to the desired size
  if(size_flag==0):
    return tf.image.resize(img, [smallestH, smallestW])
  if(size_flag==1):
    return tf.image.resize(img, [28, 35])


def predict_from_path(image_path, model, size_flag=0):
    image = tf.io.read_file(image_path)
    image = decode_img(image, size_flag)
    image = image/255.0
    image = tf.expand_dims(image, 0)
    
    
    prediction = model.predict(image)
    raw_pred = prediction
    #return raw_pred
    prediction = tf.argmax(prediction[0], axis=-1)
    
    return prediction

#PREDICTION
def get_seq_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  one_hot = file_path[0] == LABELS_CATEGORY
  one_hot = tf.cast(one_hot, dtype=tf.float32)
  
  

  return one_hot
def create_pred_dataset(data):
    
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.map(get_seq_label)
    ds = ds.batch(8, drop_remainder=True)
    
    ds = ds.batch(1, drop_remainder=True)
    
    return ds


def predict_rnn(data, model):
    prediction = model.predict(data, batch_size=1)
    
    return prediction


cnn_model = models.load_model('trained_nns/cnn_progress3')
units_model = models.load_model('trained_nns/cnn_units')
tens_model = models.load_model('trained_nns/cnn_tens')
minutes_model = models.load_model('trained_nns/minutes_units')
minutesT_model = models.load_model('trained_nns/minutes_tens')

rnn_model = models.load_model('trained_nns/rnn')


LABELS_CATEGORY = listdir('Labeled_Img')
LABELS_UNITS = listdir('num_data/num_units')
LABELS_TENS = listdir('num_data/num_tens')

#Player 1
STARTING_HEIGHT = 60

#Player 2
#STARTING_HEIGHT = 143
STARTING_WIDTH = 55

PRODUCTION_HEIGHT = 50
PRODUCTION_WIDTH = 47

NEXT_PRODUCTION= 16 + PRODUCTION_WIDTH

pred = None
prediction_rnn = None
count = 0
previous_production = []
sequence = []
#sequence.append(['None', 0, 0])
while True:
    current_production = []
    myScreenshot = pyautogui.screenshot()
    myScreenshot.save(r'games_img'+GAME_PATH+'screenshot'+str(count)+'.png')
    
    img = mpimg.imread('games_img'+GAME_PATH+'screenshot'+str(count)+'.png')
    for slice_production in range(9):
        predicted = []
        looping_width = STARTING_WIDTH + NEXT_PRODUCTION * slice_production
        
        #imgplot = plt.imshow(img[STARTING_HEIGHT:STARTING_HEIGHT+PRODUCTION_HEIGHT, looping_width:looping_width+PRODUCTION_WIDTH])
        plt.imsave('aux_prediction_img/predicted.png', img[STARTING_HEIGHT:STARTING_HEIGHT+PRODUCTION_HEIGHT, looping_width:looping_width+PRODUCTION_WIDTH])
        prediction = predict_from_path('aux_prediction_img/predicted.png', cnn_model)
        category = LABELS_CATEGORY[prediction]
        if(category == 'None'):
            break
        predicted.append(category)
        
        
        prediction = predict_from_path('aux_prediction_img/predicted.png', units_model)
        units = int(LABELS_UNITS[prediction])
        
        prediction = predict_from_path('aux_prediction_img/predicted.png', tens_model)
        tens = int(LABELS_TENS[prediction])
        
        predicted.append(units+tens)
        
        plt.imsave('aux_prediction_img/predicted.png', img[770:798, 0:35])        
        prediction = predict_from_path('aux_prediction_img/predicted.png', minutes_model, 1)
        m_units = int(LABELS_UNITS[prediction])
        
        prediction = predict_from_path('aux_prediction_img/predicted.png', minutesT_model, 1)
        m_tens = int(LABELS_TENS[prediction])        
        
        predicted.append(m_units+m_tens)
        
        current_production.append(predicted)
        #current_production.append(category)
        #plt.show()
    #print(current_production)
    #print(set(current_production)==set(previous_production))
    
    if len(sequence)>8:
        pred_ds = create_pred_dataset(np.array(sequence[-8:]))
        prediction_rnn  = predict_rnn(pred_ds, rnn_model)
    #for i in prediction:
    #    pred = tf.argmax(i[:-2])
    #    tf.print(LABELS_CATEGORY[pred], tf.cast(i[-2], dtype=tf.int32), tf.cast(i[-1], dtype=tf.int32))
    #    print()
     
    #print(prediction)
    #pred = tf.argmax(prediction)
    
    if prediction_rnn is not None:
        pred = tf.argmax(prediction_rnn[0])
    if pred!=None:
        tf.print("\n\nPrediction: ", LABELS_CATEGORY[int(pred)], "\n")
    
    previous_aux = [i[0] for i in previous_production]
    [sequence.append(item) for item in current_production if item[0] not in previous_aux]
    for item_i in current_production:
        for item_j in previous_production:
            if item_i[0]==item_j[0]:
                if item_i[1]>item_j[1]:
                    sequence.append([item_i[0], item_i[1]-item_j[1], item_i[2]])
            
    print(sequence)
    print()
    previous_production = current_production.copy()
    count+=1
    time.sleep(1.5)
    #os.system('cls')
    
    #Saving the csv
    df = pd.DataFrame(sequence, columns=["Production", "Units", "Minutes"])
    df.to_csv("Sequences_Data/IEM202Extra.csv", index=False)