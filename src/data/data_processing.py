#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 17:20:06 2018

@author: chrispedder

A routine to crop sections from the images of different manuscripts in the two
datasets to the same size, and with the same magnification, so that the average
script size doesn't create a feature that the neural networks can learn.

Reading the data description of the CLaMM dataset, we find that the images
are 150mm*100mm, so we need to take similar-sized crops from our new target
data. Looking at the bar on the left, we find that 6000px =(341-47) = 294mm
So 1mm = 20.41px. We therefore need to crop 3062 * 2041px from the original
However, to not give away too much, we need to make this crop a little
random. Looking at the test images, 1) Their heights vary by around 100px
AFTER downsampling, so around 170px BEFORE downsampling. 2) Their widths
vary by proportionately less, around 65px AFTER, so 110px BEFORE.

We define a crop function below which achieves precisely this.

To run this routine, call something like `python -m src.data.data_processing
--thow_input_path data/raw/MS157/ --thow_output_path data/external/thow_out
--clamm_input_path data/raw/ICDAR2017_CLaMM_Training/
--clamm_output_path data/external/clamm_out`

The four command line args given here are all required.
"""
import numpy as np
import scipy.io
import random
import scipy.ndimage
import glob
import os
import argparse

from PIL import Image
from random import randint
from typing import List

# helper function to clean up file list for scraped THoW filenames
def clean_THoW_file_list(file_list: List):
    # clean out folio views etc, whose filenames start with a letter
    # rather than a number
    cleaned_THoW_list = [element for element in file_list if not
                         element[-5].isalpha()]

    return cleaned_THoW_list

class ImageProcessor(object):
    def __init__(self, args):
        # Crop images from point CORNER, to size given by DIM
        self.CORNER = [1000,1400]
        self.DIM = [3062,2041]
        # To match the training data, we need to downsample images by a
        # factor of 1.7
        self.SCALE = 1.7

        # Set size of training tiles here
        self.IM_HEIGHT = 300
        self.IM_WIDTH = 300

        # Set random seed to get the same train-test split when run
        self.SEED = 42
        random.seed(self.SEED)

        self.args = args

    def read_raw_from_dir(self, filename):
        """
        Define function to read bytes directly from tar by filename.
        """
        x = Image.open(filename)
        x = x.convert('L')
        # makes it greyscale - CLaMM data is already grayscale
        y = np.asarray(x.getdata(), dtype='uint8')
        return y.reshape((x.size[1], x.size[0]))

    def image_oc_crop(self, img):
        """
        Makes a crop of an img, with coordinates of the top left corner
        top_left_pt and of side lengths "dimensions" using numpy slicing.
        """
        lh, lw = self.CORNER
        dim_x, dim_y = self.DIM
        cropped_img = img[lh:lh+dim_x,lw:lw+dim_y]
        return cropped_img

    def resample_image(self, img):
        """
        Resample scraped images to make them a similar number of pixels to
        CLaMM dataset images.
        """
        # retain a single image channel, use cubic splines for resampling
        resampled = scipy.ndimage.zoom(img, 1/self.SCALE, order=3)
        output = resampled.astype('uint8')
        return output


    def prepare_raw_bytes_for_model(self, input_path):
        """
        Put everything together into one function to read, crop & scale data
        """
        input_image = self.read_raw_from_dir(input_path)
        cropped_input = self.image_oc_crop(input_image)
        img = self.resample_image(cropped_input)
        return img

    def tile_crop(self, array):
        """
        function to crop tile_height by tile_width sections from the original
        cropped files.
        """
        array_height, array_width = array.shape

        height_tiles_number = array_height//self.IM_HEIGHT
        width_tiles_number = array_width//self.IM_WIDTH

        tile_list = []
        for i in range(height_tiles_number):
            for j in range(width_tiles_number):
                new_tile = array[i * self.IM_HEIGHT: (i + 1) * self.IM_HEIGHT,
                                 j * self.IM_WIDTH: (j + 1)* self.IM_WIDTH]
                tile_list.append(new_tile)

        return tile_list

    def write_input_data_to_jpg(self, input_path, output_path, THOW=False):
        """
        Read files, process and write out processed files to an external folder,
        defined by the argparse args
        """
        counter = 0
        file_suffix = '*.jpg' if THOW else '*.tif'
        file_name = 'THOW' if THOW else 'CLaMM'

        # get list of files in the raw data directory
        input_files_list = sorted(glob.glob(input_path + file_suffix))
        if THOW:
            input_files_list = clean_THoW_file_list(input_files_list)
        else:
            input_files_list = input_files_list[:500]

        #check output directory exists, if not create it
        if not os.path.exists(output_path):
                os.mkdir(output_path)

        for element in input_files_list:
            image = self.prepare_raw_bytes_for_model(element)
            new_tile_list = self.tile_crop(image)

            for i, tile in enumerate(new_tile_list):
                # define file names for training example
                tile_file_name = os.path.join(
                    output_path,
                    file_name + str(counter + i) + ".jpg")
                # write three copies of the grayscale image to three separate
                # layers as the VGG16 net expects an RGB input
                tensorized = np.dstack([tile] * 3)
                # create image from tensorized array
                im = Image.fromarray(tensorized)
                # save to path specified in arguments
                im.save(tile_file_name)
                print(
                    "Tile with name {} written to disk".format(tile_file_name))

            counter += len(new_tile_list)
            print("So far {} files written".format(counter))

        print("File writing completed")

    def process_all_files(self):
        print(f'Reading data from {self.args.thow_input_path}, writing to\
            {self.args.thow_output_path}')
        self.write_input_data_to_jpg(self.args.thow_input_path,
                                     self.args.thow_output_path,
                                     THOW=True)
        print(f'Reading data from {self.args.clamm_input_path}, writing to\
            {self.args.clamm_output_path}')
        self.write_input_data_to_jpg(self.args.clamm_input_path,
                                     self.args.clamm_output_path)

        print('All files processed and written to file')

def parse_args():
    parser = argparse.ArgumentParser(description='Command line options for '
        'processing the data files needed to train the model.')
    parser.add_argument('--thow_input_path', type=str, required=True,
        help='give the path to the THOW raw files')
    parser.add_argument('--thow_output_path', type=str, required=True,
        help='path to where we should write the processed THOW tile files')
    parser.add_argument('--clamm_input_path', type=str, required=True,
        help='give the path to the CLaMM raw files')
    parser.add_argument('--clamm_output_path', type=str, required=True,
        help='path to where we should write the processed CLaMM tile files')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    processor = ImageProcessor(args)
    processor.process_all_files()
