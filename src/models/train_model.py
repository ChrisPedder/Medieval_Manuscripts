#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 11:07:05 2018

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
from keras.callbacks import ModelCheckpoint, TensorBoard

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.applications.vgg16 import VGG16 

from keras.preprocessing.image import ImageDataGenerator

IM_HEIGHT = 300
IM_WIDTH = 300

# Set random seed to get the same train-test split when run
SEED = 42
random.seed(SEED)

# User-defined model hyperparameters
epochs = 50
epochs2 = 10
batch_size = 20
l1_norm_weight = 0.0001

### Do train/test split of dataset for training purposes, retain specific
### split used for future use/investigation

def generate_train_test_split_files(train_test_split, set_size):
    # Set size of train-test split
    split = round((1-train_test_split) * set_size)

    # get list of files in the raw data directory
    root = Path.cwd()
    MS_folder = str(root.parent.parent) + '/' + 'data/interim/MS157' + '/'
    CLaMM_folder = str(root.parent.parent) + '/' + 'data/interim/CLaMM' + '/'
    
    # take random sample of size SET_SIZE from examples
    MS_sample_list = random.sample(glob.glob(MS_folder + '/*'), set_size)
    CLaMM_sample_list = random.sample(glob.glob(CLaMM_folder + '/*'),
                                     set_size)
        
    MS_tr_files = []
    CLaMM_tr_files = []
    for i in range(split):
        MS_tr_files.append(MS_sample_list[i])
        CLaMM_tr_files.append(CLaMM_sample_list[i])

    MS_te_files = []
    CLaMM_te_files = []
    for i in range(split,set_size):
        MS_te_files.append(MS_sample_list[i])
        CLaMM_te_files.append(CLaMM_sample_list[i])

    return [MS_tr_files, MS_te_files, CLaMM_tr_files, CLaMM_te_files]

def add_path(string):
    """
    Helper function for following routine to generate list of train/test
    target directories required by Keras...
    """
    root = Path.cwd()
    top_path = str(root.parent.parent)
    return top_path + '/' + string + '/'

def generate_target_train_test_directories(set_size):
    """
    Create set of routines for making a list of the target directories for
    copied train/test split files    
    """

    target_list = ['data/processed/set_size_' + str(set_size),\
                   'data/processed/set_size_' + str(set_size) + '/train',\
                   'data/processed/set_size_' + str(set_size) + '/test',\
                   'data/processed/set_size_' + str(set_size) + '/train/' + 'MS157',\
                   'data/processed/set_size_' + str(set_size) + '/test/' + 'MS157',\
                   'data/processed/set_size_' + str(set_size) + '/train/' + 'CLaMM',\
                   'data/processed/set_size_' + str(set_size) + '/test/' + 'CLaMM']
    
    train_test_directories = []
    for extension in target_list:
        train_test_directories.append(add_path(extension))

    return train_test_directories

def do_train_test_split(split, set_size):
    """
    Do train-test split of data into subfolders required for Keras 
    retraining format.
    """
        
    file_locations = generate_train_test_split_files(split, set_size)
    
    # generate list of target directories to copy files to
    path_list = generate_target_train_test_directories(set_size)

    if all(os.path.exists(x) for x in path_list):
        print('Train test split already exists for this set size. Using existing split...')
        return

    else:
        #check chosen directory exists, if not create it
        for path in path_list:
            os.mkdir(path)

        # copy files to train and test directories
        for i in range(4):
            for filename in file_locations[i]:
                shutil.copy2(filename, path_list[i+3])
                print("File named {} copied to directory {}".format(filename, 
                      file_locations[i]))

        return

# Keep record of all hyperparameters in JSON file. First write JSON from input
# parameters, then read in to set values for deep learning to ensure no
# errors/contamination

# Write model hyperparameters to file
def write_JSON_file(path, filename, data):
    filepath = path + '/' + filename + '.json'
    with open(filepath, 'w') as fp:
        json.dump(data, fp)
        
def get_date_string():
    now = datetime.datetime.now()
    year = now.year
    month = now.month
    day = now.day
    date_string = str(day) + '_' + str(month) + '_' + str(year)
    return date_string

def create_log_weights_file_names(set_size):
    
    date = get_date_string()
    
    log_file_name = 'logfile' + '_setsize_' + str(set_size) + '_e1_' +\
    str(epochs) + '_e2_' + str(epochs2) + '_bs_' + str(batch_size) +\
    '_l1_' + str(l1_norm_weight) + '_' + date
    
    weight_file_name = 'weights' + '_setsize_' + str(set_size) + '_e1_' +\
    str(epochs) + '_e2_' + str(epochs2) + '_bs_' + str(batch_size) +\
    '_l1_' + str(l1_norm_weight) + '_' + date
    
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
    hyperparams_dictionary['epochs'] = epochs
    hyperparams_dictionary['epochs_2'] = epochs2
    hyperparams_dictionary['batch_size'] = batch_size
    hyperparams_dictionary['l1_norm_weight'] = l1_norm_weight
    
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
    
    epochs = data['epochs']
    epochs2 = data['epochs_2']
    batch_size = data['batch_size']
    l1_norm_weight = data['l1_norm_weight']
    
    return epochs, epochs2, batch_size, l1_norm_weight
    
### Start deep learning

# Save files containing the feature map data for training and test sets from running VGG16
def get_training_validation_features(train_test_split, set_size):
    
    nb_training_features = round((1-train_test_split) * set_size)
    
    nb_test_features = set_size - nb_training_features
    
    # path to the model weights files.
    root_path = Path.cwd()

    train_features_path = str(root_path.parent.parent) + '/data/processed/' + 'set_size_'\
    + str(set_size) + '/train/'

    test_features_path = str(root_path.parent.parent) + '/data/processed/' + 'set_size_'\
    + str(set_size) + '/test/'
    
    return nb_training_features, nb_test_features, train_features_path, test_features_path

def save_bottleneck_features(split, set_size, batch_size):
        
    # Data augmentation using affine transformations etc.
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network with false colour start 
    model = applications.VGG16(weights='imagenet', include_top=False)
    
    nb_training_features, nb_test_features, train_features_path,\
    test_features_path = get_training_validation_features(split, set_size)
    
    if not any(fname.endswith('.npy') for fname in os.listdir(train_features_path)):
        # Augmentation generator using flow_from_directory
        generator = datagen.flow_from_directory(
                train_features_path,
                target_size=(IM_HEIGHT, IM_WIDTH),
                batch_size=batch_size,
                class_mode=None,
                shuffle=False)
        bottleneck_features_train = model.predict_generator(
                generator, nb_training_features // batch_size)
    
        # Save bottleneck features for training images so we don't have to rerun VGG16
        np.save(open(train_features_path + 'bottleneck_features_train','wb'),
                bottleneck_features_train)

    else:
        print('Bottleneck training set features already saved for this sample\
              size. Using previously created version')
        
    if not any(fname.endswith('.npy') for fname in os.listdir(test_features_path)):

        generator = datagen.flow_from_directory(
                test_features_path,
                target_size=(IM_HEIGHT, IM_WIDTH),
                batch_size=batch_size,
                class_mode=None,
                shuffle=False) 

        bottleneck_features_test = model.predict_generator(
                generator, nb_test_features // batch_size)
        
        # Save bottleneck features for training images so we don't have to rerun VGG16
        np.save(open(test_features_path + 'bottleneck_features_test','wb'),
                bottleneck_features_test)

    else:
        print('Bottleneck test set features already saved for this sample\
              size. Using previously created version')


# Build small classifier network to sit on top of VGG16

def build_classifier_model(data_shape, l1_norm_weight):
    model = Sequential()
    model.add(Flatten(input_shape=data_shape))
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l1(l1_norm_weight)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Train small discriminator model on the feature maps from VGG16 saved above
def train_top_model(split, set_size, epochs, batch_size, l1_norm_weight):
    
    nb_training_features, nb_test_features, train_features_path,\
    test_features_path = get_training_validation_features(split, set_size)
    
    log_path, weights_path = create_log_weights_file_paths()
        
    _, weights_file_name = create_log_weights_file_names(set_size)

    train_data = np.load(open(train_features_path + 'bottleneck_features_train','rb'))
    train_labels = np.array(
        [0] * (nb_training_features // 2) + [1] * (nb_training_features // 2))

    validation_data = np.load(open(test_features_path + 'bottleneck_features_test','rb'))
    validation_labels = np.array(
        [0] * (nb_test_features // 2) + [1] * (nb_test_features // 2))

    model = build_classifier_model(train_data.shape[1:], l1_norm_weight)

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    checkpointer = keras.callbacks.ModelCheckpoint(weights_path + weights_file_name, 
                                                   monitor='val_acc', 
                                                   verbose=1, save_best_only=True, 
                                                   save_weights_only=True)

    tensorboard = TensorBoard(log_dir = log_path + '/tensorboard',
                              histogram_freq = 1, 
                              write_graph = True, 
                              write_images = True)


    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
#              callbacks = [checkpointer])
              callbacks = [checkpointer, tensorboard])


def retrain_vgg_network():    
    input_tensor = Input(shape=(300,300,3))
    base_model = VGG16(weights='imagenet',include_top= False,input_tensor=input_tensor)
    print('Model loaded.')

    top_model = build_classifier_model(base_model.output_shape[1:])
    top_model.load_weights(top_model_weights_path)

    model = Model(input= base_model.input, output= top_model(base_model.output))

    for layer in model.layers[:15]:
        layer.trainable = False

    print(model.summary())

## compile the model with a SGD/momentum optimizer
## and a very slow learning rate.

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    
    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary') 

    validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary') 

    checkpointer = ModelCheckpoint(filepath='check_weights.hdf5', verbose=1, 
                                   monitor='val_acc', save_best_only=True)

    # fine-tune the model
    model.fit_generator(
            train_generator,
            steps_per_epoch = nb_train_samples//batch_size,
            epochs=epochs2,
            validation_data = validation_generator,
            verbose=1,
            callbacks = [checkpointer])

    return

def Main():

    parser = argparse.ArgumentParser()
    parser.add_argument('split_size', help = 'The size of the holdout test set used\
                        to quantify model performance as a fraction of the total\
                        size of the training set', type = float)

    parser.add_argument('set_size', help = 'The number of images from each\
                        training set to use', type = int)


    args = parser.parse_args()

    train_test_split = args.split_size
    set_size = args.set_size
    
    do_train_test_split(train_test_split, set_size)
    
    write_hyperparameters_to_json(set_size)
    
    epochs, epochs2, batch_size, l1_norm_weight = read_hyperparameters_from_json(set_size)

#    save_bottleneck_features(train_test_split, set_size, batch_size)
    
    train_top_model(train_test_split, set_size, epochs, batch_size, l1_norm_weight)

    
    return

if __name__ == '__main__':
    Main()

