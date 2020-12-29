"""
A routine for writing TFRecords files of the image embeddings and labels
for the datasets used to train the top models.
"""

import os
import argparse
import tensorflow as tf
import numpy as np

from .Predictors import predictors_options

# helper functions for converting float and int features to tf.train compatible
# features

def _floats_feature(value):
   return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class TFRecordsWriter(object):
    def __init__(self, args):
        self.IM_HEIGHT = 300
        self.IM_WIDTH = 300
        self.SEED = 42

        self.batch_size = args.batch_size
        self.train_test_split = args.train_test_split
        self.data_dir = args.data_dir
        self.predictor = predictors_options[args.embedding_model](args)
        self.datasets = self.get_tf_datasets()

        self.tfrecords_foldername = os.path.join(
            args.data_dir,
            self.predictor.model.name + '_tfrecords')
        if not os.path.isdir(self.tfrecords_foldername):
            os.mkdir(self.tfrecords_foldername)

    def normalize_images(self, dataset):
        normalization_layer = \
            tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
        normalized_ds = dataset.map(lambda x, y: (normalization_layer(x), y))
        return normalized_ds

    def get_tf_datasets(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=(1-self.train_test_split),
            subset="training",
            seed=self.SEED,
            image_size=(self.IM_HEIGHT, self.IM_WIDTH),
            batch_size=self.batch_size)

        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=(1-self.train_test_split),
            subset="validation",
            seed=self.SEED,
            image_size=(self.IM_HEIGHT, self.IM_WIDTH),
            batch_size=self.batch_size)

        print(train_ds.class_names)

        normalized_train = self.normalize_images(train_ds)
        normalized_test = self.normalize_images(test_ds)

        return {'train': normalized_train,
                'test': normalized_test}

    def write_to_tf_records(self):
        for name, dataset in self.datasets.items():
            print(f'Storing dataset {name} to TFRecords')
            tfrecords_filename = os.path.join(self.tfrecords_foldername, name)
            writer = tf.io.TFRecordWriter(tfrecords_filename)
            i = 0
            for batch, label in dataset:
                print(f'Converting batch {i} to TFRecords')
                features = self.predictor.predict(batch).tolist()
                labels = label.numpy().tolist()

                for embedding, label in zip(features, labels):

                    embed_tensor = tf.io.serialize_tensor(embedding)
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'embedding': _bytes_feature(embed_tensor),
                                'labels': _int64_feature(label),
                            }
                        )
                    )

                    writer.write(example.SerializeToString())
                    i += 1

            writer.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='The size of the batches to use '
                        'when writing data to TFRecords files', type=int,
                        default=128)
    parser.add_argument('--train_test_split', help='The size of the train set '
                        'as a proportion of all files', type=float, default=0.8)
    parser.add_argument('--data_dir', help='Path to the data tiles',
                        type=str, required=True)
    parser.add_argument('--embedding_model', help='which embedding model to '
                        'use to generate embeddings', type=str, default='vgg16')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    tfr = TFRecordsWriter(args)
    tfr.write_to_tf_records()
    print("finished")
