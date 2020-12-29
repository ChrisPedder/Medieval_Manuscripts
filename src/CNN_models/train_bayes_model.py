#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 11:07:05 2018

@author: chrispedder
"""

import numpy as np
import os
import argparse
import json
import tensorflow as tf
import tensorflow_probability as tfp

from abc import ABC, abstractmethod
from datetime import datetime

from .TFRecordsReader import TFRecordsReader
from ..data.Predictors import (
    predictors_options, VGG16Predictor, embedding_sizes)
from .train_model import (jsonify, DeterministicModel, parse_args)

tfd = tfp.distributions

class ProbabilisticModel(DeterministicModel):

    def build_model(self):
        """Creates a Keras model using the LeNet-5 architecture.
        Returns:
          model: Compiled Keras model.
        """
        # KL divergence weighted by the number of training samples, using
        # lambda function to pass as input to the kernel_divergence_fn on
        # flipout layers.
        kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /
            tf.cast(15000, dtype=tf.float32))

        model = tf.keras.models.Sequential([
          tf.keras.Input(shape=(self.embed_size,)),
          tfp.layers.DenseFlipout(
              self.hidden_size, kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.relu),
          tf.keras.layers.Dropout(self.dropout),
          tfp.layers.DenseFlipout(
              1, kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.softmax)
        ])
        return model

    def train(self):
        self.logs_folder = self.create_model_training_folder()
        # Model compilation.
        optimizer = tf.keras.optimizers.Adam(lr=self.args.learning_rate)
        # We use the categorical_crossentropy loss since the MNIST dataset contains
        # ten labels. The Keras API will then automatically add the
        # Kullback-Leibler divergence (contained on the individual layers of
        # the model), to the cross entropy loss, effectively
        # calcuating the (negated) Evidence Lower Bound Loss (ELBO)
        self.model.compile(optimizer, loss='binary_crossentropy',
                      metrics=['accuracy'],
                      experimental_run_tf_function=False)

        checkpointer = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self.logs_folder, 'checkpoints'),
            monitor='val_accuracy',
            verbose=1, save_best_only=True,
            save_weights_only=True)

        self.model.fit(
            x=self.datasets['train'],
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=self.datasets['test'],
            callbacks = [checkpointer])

        self.write_config_to_json()

if __name__ == '__main__':
    args = parse_args()
    model = ProbabilisticModel(args)
    model.train()
