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
import sys
sys.path.append("..")
import argparse

# logging libraries
import json

# Machine learning libraries
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tf.enable_v2_behavior()
tfd = tfp.distributions

from .train_model import(generate_train_test_split_files, do_train_test_split,
    write_JSON_file, get_date_string, create_log_weights_file_names,
    create_log_weights_file_paths, get_training_validation_features,
    save_bottleneck_features, )
IM_HEIGHT = 300
IM_WIDTH = 300

# Set random seed to get the same train-test split when run
SEED = 42
random.seed(SEED)

def get_training_hyperparams():

    parser = argparse.ArgumentParser()
    parser.add_argument('split_size', help = 'The size of the holdout test set used\
                        to quantify model performance as a fraction of the total\
                        size of the training set', type = float)

    parser.add_argument('set_size', help = 'The number of images from each\
                        training set to use. NB set_size * split_size should\
                        be chosen to be exactly divisible by two!', type = int)

    parser.add_argument('epochs', help='Number of epochs to train the \
                        probabilistic model for', type=int, default=10)

    parser.add_argument('batch_size', help='Size of batches to train the model\
                        with', type=int, default=40)


    args = parser.parse_args()
    return args

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

def build_probabilistic_classifier_model(data_shape):
    kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /
                            tf.cast(num_samples, dtype=tf.float32))

    input_layer = tf.keras.layers.Input(shape=data_shape)
    dense_layer = tfp.layers.DenseFlipout(
      units=1,
      activation='sigmoid',
      kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
      bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
      kernel_divergence_fn=kl_divergence_function)(input_layer)

    # Model compilation.
    model = tf.keras.Model(inputs=input_layer, outputs=dense_layer)
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_top_model(split, nb_samples, epochs, batch_size, l1_norm_weight):

    nb_training_features, nb_test_features, train_features_path,\
    test_features_path = get_training_validation_features(split, nb_samples)

    log_path, weights_path = create_log_weights_file_paths()

    _, weights_file_name = create_log_weights_file_names(nb_samples//2)

    train_data = np.load(
        open(train_features_path + 'bottleneck_features_train','rb'))
    train_labels = np.array(
        [0] * (nb_training_features // 2) + [1] * (nb_training_features // 2))

    validation_data = np.load(
        open(test_features_path + 'bottleneck_features_test','rb'))
    validation_labels = np.array(
        [0] * (nb_test_features // 2) + [1] * (nb_test_features // 2))

    model = build_classifier_model(train_data.shape[1:], l1_norm_weight)

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    print(nb_training_features, nb_test_features)

    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        weights_path + weights_file_name,
        monitor='val_acc',
        verbose=1, save_best_only=True,
        save_weights_only=True)

    model.fit(
        train_data, train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(validation_data, validation_labels),
        callbacks = [checkpointer])

def Main():

    args = get_training_hyperparams()
    train_test_split = args.split_size
    set_size = args.set_size

    do_train_test_split(args.split_size, args.set_size)

    write_hyperparameters_to_json(args.set_size)

    nb_samples, epochs, epochs2, batch_size, l1_norm_weight\
    = read_hyperparameters_from_json(args.set_size)

    print(get_training_validation_features(args.split_size, nb_samples))

    save_bottleneck_features(args.split_size, nb_samples, batch_size)

    train_top_model(args.split_size, nb_samples, args.epochs, batch_size)

    return

if __name__ == '__main__':
    Main()
