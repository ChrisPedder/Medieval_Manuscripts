"""
A routine for reading TFRecords files of the image embeddings and labels
generated for the datasets used to train the top models.
"""
import os
import argparse
import tensorflow as tf
import numpy as np

class TFRecordsReader(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.data_dir = args.data_dir
        self.tfrecords_foldername = os.path.join(
            args.data_dir,
            args.embedding_model + '_tfrecords')
        self.datasets = self.read_from_tf_records()

    def read_from_tf_records(self):
        dataset_dict = {}
        for name in ['train', 'test']:
            raw_image_dataset = tf.data.TFRecordDataset(
                os.path.join(self.tfrecords_foldername, name))
            parsed_image_dataset = raw_image_dataset.map(
                self._parse_image_function)
            dataset_dict[name] = parsed_image_dataset.batch(
                self.batch_size)

        return dataset_dict

    def _parse_image_function(self, example_proto):
        # Create a dictionary describing the features.
        image_feature_description = {
            'embedding': tf.io.FixedLenFeature([], tf.string),
            'labels': tf.io.FixedLenFeature([], tf.int64),
        }
        # Parse the input tf.train.Example proto using the dictionary above.
        example = tf.io.parse_single_example(
            example_proto, image_feature_description)
        embedding = tf.io.parse_tensor(
            example['embedding'], out_type = tf.float32)
        return embedding, example['labels']


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
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    tfr = TFRecordsReader(args)
    tds = tfr.datasets['train']
    print(list(tds.as_numpy_iterator()))
