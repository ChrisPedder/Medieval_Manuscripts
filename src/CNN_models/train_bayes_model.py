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
import tensorflow_probability as tfp

from abc import ABC, abstractmethod
from datetime import datetime

from .TFRecordsReader import TFRecordsReader
from ..data.Predictors import (
    predictors_options, VGG16Predictor, embedding_sizes)
from .train_model import (jsonify, DeterministicModel, parse_args)

tfd = tfp.distributions
tfpl = tfp.layers

def nll(y_true, y_pred):
    """
    This function should return the negative log-likelihood of each sample
    in y_true given the predicted distribution y_pred. If y_true is of shape
    [B, E] and y_pred has batch shape [B] and event_shape [E], the output
    should be a Tensor of shape [B].
    """
    return -y_pred.log_prob(y_true)

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

def one_hot_encode(image, label):
    """One hot encodes labels: `int64` -> `[int64]`."""
    return image, tf.one_hot(label, 2)

class ProbabilisticModel(DeterministicModel):

    def get_train_test_datasets(self):
        reader = TFRecordsReader(self.args)
        for key, value in reader.datasets.items():
            value = value.map(
                normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
            value = value.map(
                one_hot_encode, num_parallel_calls=tf.data.AUTOTUNE)
            reader.datasets[key] = value
        return reader.datasets

    def build_model(self):
        """Creates a Keras model using the LeNet-5 architecture.
        Returns:
          model: Compiled Keras model.
        """

        model = tf.keras.models.Sequential([
          tf.keras.Input(shape=(self.embed_size,)),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dropout(self.dropout),
          tf.keras.layers.Dense(tfpl.OneHotCategorical.params_size(2)),
          tfpl.OneHotCategorical(2, convert_to_tensor_fn=tfd.Distribution.mode)
        ])

        return model

    def train(self):
        self.logs_folder = self.create_model_training_folder()
        # Model compilation.
        optimizer = tf.keras.optimizers.RMSprop()
        # We use the categorical_crossentropy loss since the MNIST dataset contains
        # ten labels. The Keras API will then automatically add the
        # Kullback-Leibler divergence (contained on the individual layers of
        # the model), to the cross entropy loss, effectively
        # calcuating the (negated) Evidence Lower Bound Loss (ELBO)
        self.model.compile(optimizer, loss=nll,
                      metrics=['accuracy'],
                      experimental_run_tf_function=False)
        print(self.model.summary())
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


class CompleteProbabilisticModel(DeterministicModel):
    # Define the prior weight distribution as Normal of mean=0 and stddev=1.
    # Note that, in this example, the prior distribution is not trainable,
    # as we fix its parameters.
    def prior(self, kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        prior_model = tf.keras.Sequential(
            [
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.MultivariateNormalDiag(
                        loc=tf.zeros(n), scale_diag=tf.ones(n)
                    )
                )
            ]
        )
        return prior_model

    # Define variational posterior weight distribution as multivariate Gaussian.
    # Note that the learnable parameters for this distribution are the means,
    # variances, and covariances.
    def posterior(self, kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        posterior_model = tf.keras.Sequential(
            [
                tfp.layers.VariableLayer(
                    tfp.layers.MultivariateNormalTriL.params_size(n),
                    dtype=dtype
                ),
                tfp.layers.MultivariateNormalTriL(n),
            ]
        )
        return posterior_model

    # def compute_train_size(self):
    #     count = 0
    #     for entry in self.datasets['train']:
    #         count += self.batch_size
    #     print(f"Train set size = {count}")
    #     return count


    def build_model(self):
        """Creates a Keras model using probabilistic layers.
        Returns:
          model: Keras model (needs to be compiled)
        """
        inputs = tf.keras.layers.Input(shape=(self.embed_size,), dtype=tf.float32)
        features = tf.keras.layers.BatchNormalization()(inputs)
        # Create hidden layer with weight uncertainty using the
        # DenseVariational layer.
        features = tfp.layers.DenseVariational(
            units=self.hidden_size,
            make_prior_fn=self.prior,
            make_posterior_fn=self.posterior,
            # kl_weight=1/self.compute_train_size(),
            # kl_weight=1/4800,
            activation="sigmoid",
        )(inputs)

        # Create a probabilistic output, and use the `Dense` layer
        # to produce the parameters of the Bernouilli distribution
        # (we use Bernouilli since we're doing a binary classification).
        # We set units=1 to learn the distribution param.
        distribution_params = tf.keras.layers.Dense(units=1)(features)
        outputs = tfp.layers.IndependentBernoulli(1)(distribution_params)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def negative_loglikelihood(self, targets, estimated_distribution):
        return -estimated_distribution.log_prob(targets)

    def train(self):
        self.logs_folder = self.create_model_training_folder()
        # Model compilation.
        optimizer = tf.keras.optimizers.Adam(lr=self.args.learning_rate)
        # self.model.compile(optimizer, loss='binary_crossentropy',
        #               metrics=['accuracy'])
        self.model.compile(optimizer, loss=self.negative_loglikelihood,
                      metrics=['accuracy'])

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
    # model = CompleteProbabilisticModel(args)
    # model.train()

    model = ProbabilisticModel(args)
    model.train()
