#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:11:51 2020

@author: msabry
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from sklearn.metrics import recall_score, precision_score, f1_score
import efficientnet.tfkeras as efn 


###METRICS
def recall_macro(y_true, y_pred):
    return recall_score(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1), average = 'macro', zero_division=0)
    
def tf_recall_macro(y_true, y_pred):
    return tf.py_function(recall_macro, inp=[y_true, y_pred], Tout=tf.float32)


def recall_micro(y_true, y_pred):
    return recall_score(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1), average = 'micro', zero_division=0)
    
def tf_recall_micro(y_true, y_pred):
    return tf.py_function(recall_micro, inp=[y_true, y_pred], Tout=tf.float32)


def precision_micro(y_true, y_pred):
    return precision_score(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1), average = 'micro', zero_division=0)
    
def tf_precision_micro(y_true, y_pred):
    return tf.py_function(precision_micro, inp=[y_true, y_pred], Tout=tf.float32)


def precision_macro(y_true, y_pred):
    return precision_score(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1), average = 'macro', zero_division=0)
    
def tf_precision_macro(y_true, y_pred):
    return tf.py_function(precision_macro, inp=[y_true, y_pred], Tout=tf.float32)


def f1_macro(y_true, y_pred):
    return f1_score(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1), average = 'macro', zero_division=0)
    
def tf_f1_macro(y_true, y_pred):
    return tf.py_function(f1_macro, inp=[y_true, y_pred], Tout=tf.float32)


def f1_micro(y_true, y_pred):
    return f1_score(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1), average = 'micro', zero_division=0)
    
def tf_f1_micro(y_true, y_pred):
    return tf.py_function(f1_micro, inp=[y_true, y_pred], Tout=tf.float32)





    
def load_model_():
    model = load_model('./model/effB1_v1.h5', custom_objects={'tf_recall_macro': tf_recall_macro,
                                                               'tf_recall_micro': tf_recall_micro,
                                                               'tf_precision_micro': tf_precision_micro,
                                                               'tf_precision_macro': tf_precision_macro,
                                                               'tf_f1_macro':tf_f1_macro,
                                                               'tf_f1_micro':tf_f1_micro})
    return model


def get_hair_dict():
    hair_color_dictionary = {0:'bald', 
                          1:'black', 
                          2:'blonde',
                          3:'brown',
                          4:'gray',
                          5:'other'}
    return hair_color_dictionary



def predict_(batch, model):
    hair_color_dictionary = get_hair_dict()
    pred = model.predict(np.array(batch).reshape(-1,224,224,3))
    
    hair_color_dict = {'label':[hair_color_dictionary[x] for x in np.argmax(pred, axis = -1)],
                       'confidence': [int(x) for x in np.int16(np.max(pred*100, axis=-1))]}  

    result = {'hair_color':hair_color_dict}
    return result




def predict_TTA(batch, model):
    limit = len(batch)//2
    hair_color_dictionary = get_hair_dict()
    pred = model.predict(np.array(batch).reshape(-1,224,224,3))
    pred_final_hair_color = (pred[0:limit] + pred[limit:])/2
    

    hair_color_dict = {'label':[hair_color_dictionary[x] for x in np.argmax(pred_final_hair_color, axis = -1)],
                       'confidence': [int(x) for x in np.int16(np.max(pred_final_hair_color*100, axis=-1))]}  

    
    result = {'hair_color':hair_color_dict}
    
    return result


