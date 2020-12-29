"""
Abstract base class and subclasses for producing different embeddings from
standard computer vision models
"""

import tensorflow as tf
import numpy as np

from abc import ABC, abstractmethod

predictors_options = {'vgg16': VGG16Predictor}

class Predictor(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.model = self.model()

    @abstractmethod
    def model(self):
        pass

    def predict(self, batch):
        return self.model.predict_on_batch(batch)

class VGG16Predictor(Predictor):
    def __init__(self, args):
        super().__init__(args)

    def model(self):
        # build the VGG16 network with false colour start
        base_model = tf.keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            pooling='max'
        )
        return base_model
