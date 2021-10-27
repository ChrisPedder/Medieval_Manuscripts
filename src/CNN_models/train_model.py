#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 11:07:05 2018

@author: chrispedder

To train the model, run from the top-level dir as:

python3 -m src.CNN_models.train_model --args ...

"""

import numpy as np
import os
import argparse
import json
import tensorflow as tf

from abc import ABC, abstractmethod
from datetime import datetime

from .TFRecordsReader import TFRecordsReader
from ..data.Predictors import (
    predictors_options, VGG16Predictor, embedding_sizes)

# Helper function for writing to JSON
def jsonify(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    return obj

class ModelTrainer(object):
    def __init__(self, args):
        self.args = args
        self.model = self.build_model()
        self.datasets = self.get_train_test_datasets()
        self.predictor = predictors_options[args.embedding_model]

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def create_model_training_folder(self):
        pass

    def safe_folder_create(self, folder):
        if not os.path.isdir(folder):
            os.mkdir(folder)

    @abstractmethod
    def get_train_test_datasets(self):
        pass

    @abstractmethod
    def write_config_to_json(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, data):
        pass

class DeterministicModel(ModelTrainer):

    def __init__(self, args):
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.dropout = args.dropout
        self.log_dir = args.log_dir
        self.embed_size = embedding_sizes[args.embedding_model]
        self.hidden_size = args.hidden_size
        super().__init__(args)

    def create_model_training_folder(self):
        # Check that top level log dir exists, if not, create it
        self.safe_folder_create(self.log_dir)

        # Next-level log dir based on date, if not already present, create it
        now = datetime.now()
        date = now.strftime("%d_%m_%Y")
        date_dir = os.path.join(self.log_dir, date)
        self.safe_folder_create(date_dir)

        # Lowest-level log dir based on numbering, if date_dir not empty,
        # check that the previous highest index was, and increment by one.
        last_index = 0
        if len(os.listdir(date_dir)) != 0:
            subfolder_list = [x[0] for x in os.walk(date_dir) if os.path.isdir(x[0])]
            last_index = max([int(x.split('_')[-1]) for x in subfolder_list[1:]])

        model_dir = os.path.join(date_dir, 'model_' + str(last_index + 1))
        self.safe_folder_create(model_dir)
        return model_dir

    def get_train_test_datasets(self):
        reader = TFRecordsReader(self.args)
        return reader.datasets

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(self.embed_size,)))
        model.add(tf.keras.layers.Dense(self.hidden_size, activation='relu'))
        model.add(tf.keras.layers.Dropout(self.dropout))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        return model

    def write_config_to_json(self):
        args_dict = vars(self.args)
        for key, value in args_dict.items():
            args_dict[key] = jsonify(value)
        json_path = os.path.join(self.logs_folder, 'config.json')
        with open(json_path, 'w') as f:
            json.dump(args_dict, f)
        print(f'Config file written to {json_path}')

    def train(self):
        self.logs_folder = self.create_model_training_folder()
        self.model.compile(optimizer='rmsprop',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        checkpointer = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self.logs_folder, 'checkpoints'),
            monitor='val_accuracy',
            verbose=1, save_best_only=True,
            save_weights_only=True)

        # tensorboard = tf.keras.callbacks.TensorBoard(
        #     log_dir = os.path.join(self.logs_folder, 'tensorboard'),
        #     histogram_freq = 1,
        #     write_graph = True,
        #     write_images = True)

        self.model.fit(
            x=self.datasets['train'],
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=self.datasets['test'],
            callbacks = [checkpointer])
            # callbacks = [checkpointer, tensorboard])

        self.write_config_to_json()

    def predict(self, data):
        args_copy = self.args
        args_copy.batch_size = 1
        pred = self.predictor(args_copy)

        outputs = []
        for entry in data:
            embedding = pred.predict(entry)
            out = self.model.predict(embedding)
            outputs.append(out)
        return outputs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='The size of the batches to use '
                        'when training the models', type=int,
                        default=32)
    parser.add_argument('--embedding_model', help='which embeddings to '
                        'use when training the model', type=str,
                        default='vgg16')
    parser.add_argument('--data_dir', help='Path to the data',
                        type=str, required=True)
    parser.add_argument('--epochs', help='How many epochs to train the model '
                        'for.', type=int, default=50)
    parser.add_argument('--dropout', help='How much dropout to apply to model ',
                        type=float, default=0.5)
    parser.add_argument('--log_dir', help='Where to save model weights and '
                        'config.', type=str, required=True)
    parser.add_argument('--hidden_size', help='What hidden sizes to use in '
                        'model.', type=int, default=256)
    parser.add_argument('--learning_rate', help='What learning rate to use in '
                        'training the model.', type=float, default=0.0001)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    model = DeterministicModel(args)
    model.train()
