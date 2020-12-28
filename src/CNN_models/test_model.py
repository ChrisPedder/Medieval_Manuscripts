#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:31:49 2018

@author: chrispedder
"""

# Scientific computing libraries
import numpy as np
import random

# File IO libraries
from pathlib import Path
import sys
import os
sys.path.append("..")
import argparse

# logging libraries
import json
import datetime
import shutil
import glob

# Machine learning libraries
import keras
from keras import applications, optimizers, regularizers
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.applications.vgg16 import VGG16 
from keras.preprocessing.image import ImageDataGenerator

import Edward as ed

IM_HEIGHT = 300
IM_WIDTH = 300

# Set random seed to get the same train-test split when run
SEED = 42
random.seed(SEED)

# User-defined model hyperparameters
EPOCHS = 150
EPOCHS2 = 10
BATCH_SIZE = 40
L1_NORM_WEIGHT = 0.0001

### Do train/test split of dataset for training purposes, retain specific
### split used for future use/investigation
def create_log_weights_file_names(set_size):
    
    date = get_date_string()
    
    log_file_name = 'logfile' + '_setsize_' + str(set_size) + '_e1_' +\
    str(EPOCHS) + '_e2_' + str(EPOCHS2) + '_bs_' + str(BATCH_SIZE) +\
    '_l1_' + str(L1_NORM_WEIGHT) + '_' + date
    
    weight_file_name = 'weights' + '_setsize_' + str(set_size) + '_e1_' +\
    str(EPOCHS) + '_e2_' + str(EPOCHS2) + '_bs_' + str(BATCH_SIZE) +\
    '_l1_' + str(L1_NORM_WEIGHT) + '_' + date
    
    return log_file_name, weight_file_name

def create_log_weights_file_paths():
    
    root_path = Path.cwd()
    
    date = get_date_string()
    
    run_path = str(root_path.parent.parent) + '/models/' + date 
    
    log_path = str(root_path.parent.parent) + '/models/' + date + '/logfiles/'
    
    weight_path = str(root_path.parent.parent) + '/models/' + date + '/weightfiles/'

    for path in [run_path, log_path, weight_path]:
        #check output directory exists, if not create it
        if not os.path.exists(path):
            os.mkdir(path)    

    return log_path, weight_path

    
def write_hyperparameters_to_json(set_size):
    hyperparams_dictionary = {}
    hyperparams_dictionary['nb_training_samples'] = 2 * set_size
    hyperparams_dictionary['epochs'] = EPOCHS
    hyperparams_dictionary['epochs_2'] = EPOCHS2
    hyperparams_dictionary['batch_size'] = BATCH_SIZE
    hyperparams_dictionary['l1_norm_weight'] = L1_NORM_WEIGHT
    
    log_path = create_log_weights_file_paths()[0]
        
    log_file_name = create_log_weights_file_names(set_size)[0]
    
    write_JSON_file(log_path, log_file_name, hyperparams_dictionary)
    
    print('Run hyperparameters written to JSON file: {}'.format(log_file_name))
    
    return
    
def read_hyperparameters_from_json(set_size):
    
    log_file_path = create_log_weights_file_paths()[0]
    
    log_file_name = create_log_weights_file_names(set_size)[0]
    
    with open(log_file_path + log_file_name + '.json') as f:
        data = json.load(f)
    
    nb_samples = data['nb_training_samples']
    epochs = data['epochs']
    epochs2 = data['epochs_2']
    batch_size = data['batch_size']
    l1_norm_weight = data['l1_norm_weight']
    
    return nb_samples, epochs, epochs2, batch_size, l1_norm_weight
    
### Start deep learning

# Save files containing the feature map data for training and test sets from running VGG16
def get_training_validation_features(train_test_split, nb_samples):
    
    nb_training_features = int((1-train_test_split) * nb_samples)
    
    nb_test_features = int(nb_samples - nb_training_features)
    
    # path to the model weights files.
    root_path = Path.cwd()

    train_features_path = str(root_path.parent.parent) + '/data/processed/' + 'set_size_'\
    + str(nb_samples//2) + '/train/'

    test_features_path = str(root_path.parent.parent) + '/data/processed/' + 'set_size_'\
    + str(nb_samples//2) + '/test/'
    
    return nb_training_features, nb_test_features, train_features_path, test_features_path

