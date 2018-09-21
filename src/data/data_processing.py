#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 17:20:06 2018

@author: chrispedder
"""

# Scientific computing libraries
import numpy as np
import scipy.io
from random import randint
import random

# Image processing libraries
from PIL import Image
import scipy.ndimage

# File IO libraries
import glob
from pathlib import Path
import os

# import argparse to handle user input
import argparse


# Define global variables used for this particular dataset

# Crop images from point CORNER, to size given by DIM
CORNER = [1000,1400]
DIM = [3062,2041]

# To match the training data, we need to downsample images by a factor of 1.7
SCALE = 1.7

# Set size of training tiles here
IM_HEIGHT = 300
IM_WIDTH = 300

# Set random seed to get the same train-test split when run
SEED = 42
random.seed(SEED)

### Reading the data description of the CLaMM dataset, we find that the images 
### are 150mm*100mm, so we need to take similar-sized crops from our new target
### data. Looking at the bar on the left, we find that 6000px =(341-47) = 294mm
### So 1mm = 20.41px. We therefore need to crop 3062 * 2041px from the original
### However, to not give away too much, we need to make this crop a little 
### random. Looking at the test images, 1) Their heights vary by around 100px 
### AFTER downsampling, so around 170px BEFORE downsampling. 2) Their widths 
### vary by proportionately less, around 65px AFTER, so 110px BEFORE.


### We define a crop function below which achieves precisely this.

### Think about absolute size of script in training.

def read_raw_from_dir(fn):
    """
    Define function to read bytes directly from tar by filename.
    """
    x = Image.open(fn)
    x = x.convert('L') 
    # makes it greyscale - CLaMM data is grayscale
    y=np.asarray(x.getdata(),dtype='uint8').reshape((x.size[1],x.size[0]))
    return y

def image_oc_crop(img, dimensions, top_left_pt): 
    """
    Makes a crop of an img, with coordinates of the top left corner top_left_pt
    and of side lengths "dimensions" using numpy slicing.
    """
    lh = top_left_pt[0]
    lw = top_left_pt[1]
    
    hdim = round(dimensions[0] + randint(-85, 85)) 
    wdim = round(dimensions[1] + randint(-55, 55))
    # randomization not really necessary
    
    cropped_img = img[lh:lh+hdim,lw:lw+wdim]
    
    return cropped_img

def resample_image(img, scale):
    """
    Resample images to make them a similar number of pixels to CLaMM dataset.
    """
    # retain a single image channel
    resampled = scipy.ndimage.zoom(img, 1/scale, order=3) 
    output = resampled.astype('uint8')
    return output


def prepare_raw_bytes_for_model(input_path):
    """
    Put everything together into one function to read, crop & scale data    
    """
    input_mat = read_raw_from_dir(input_path)
    inpt = image_oc_crop(input_mat, DIM, CORNER)
    img = resample_image(inpt,SCALE)
    return img

def tile_crop(array, tile_height, tile_width):
    """
    function to crop tile_height by tile_width sections from the original 
    cropped files.    
    """
    array_height, array_width = array.shape
    
    height_tiles_number = array_height//tile_height
    width_tiles_number = array_width//tile_width
    
    tile_list = []
    for i in range(height_tiles_number):
        for j in range(width_tiles_number):
            new_tile = array[i * tile_height: (i + 1) * tile_height, 
                             j * tile_width: (j + 1)* tile_width]
            tile_list.append(new_tile)
            
    return tile_list

def write_all_MS_data_to_jpg():
    """
    Now read MS files, process and write out processed files to an external 
    folder. Define function to write all the cropped THoW tiles to data file.    
    """
    # initialize counter to count files written
    counter = 0
    
    # get list of files in the raw data directory
    root = Path.cwd()

    input_folder = str(root.parent.parent) + '/' + 'data/raw/MS157' + '/'
    THoW_list = sorted(glob.glob(input_folder + '*.jpg'))
    
    # clean out folio views etc, whose filenames start with a letter 
    # rather than a number
    cleaned_THoW_list = [element for element in THoW_list if not 
                         element[-5].isalpha()]
    
    
    output_folder = str(root.parent.parent) + '/' + 'data/interim/MS157' + '/'
    
    #check output directory exists, if not create it
    if not os.path.exists(output_folder):
            os.mkdir(output_folder)

    for element in cleaned_THoW_list:
        
        THoW_im = prepare_raw_bytes_for_model(element)
        new_tile_list = tile_crop(THoW_im, IM_HEIGHT, IM_WIDTH)
        
        for i in range(len(new_tile_list)):
            # define file names for training example
            tile_file_name = output_folder + 'THoW'+str(counter + i)+".jpg"
            
            # write three copies of the grayscale image to three separate
            # layers as the VGG16 net expects an RGB input
            tensorized = np.dstack([new_tile_list[i]] * 3)
            
            # create image from tensorized array
            im = Image.fromarray(tensorized)
            
            # save to path specified in arguments
            im.save(tile_file_name)
            
            print("Tile with name {} written to disk".format(tile_file_name))
            
        counter += len(new_tile_list)
        
        print("So far {} files written".format(counter))       
        
    return 

def write_all_CLaMM_data_to_jpg():
    """
    Read first 500 tif files worth of cropped CLaMM tiles to data file, 
    process and write out processed files to an external folder.
    """
    # initialize counter to count files written
    counter = 0
    
    # get list of files in the raw data directory
    root = Path.cwd()
    input_folder = str(root.parent.parent) + '/' + 'data/raw/ICDAR2017_CLaMM_Training' + '/'
    clamm_list = sorted(glob.glob(input_folder + '*.tif'))[:500]
    # keep only the first 500 entries
        
    output_folder = str(root.parent.parent) + '/' + 'data/interim/CLaMM' + '/'
    
    #check output directory exists, if not create it
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for element in clamm_list:

        CLaMM_im = read_raw_from_dir(element)
        new_tile_list = tile_crop(CLaMM_im, IM_HEIGHT, IM_WIDTH)

        for i in range(len(new_tile_list)):
            # define file names for training example
            tile_file_name = output_folder + 'CLaMM'+str(counter + i)+'.jpg'

            # write three copies of the grayscale image to three separate
            # layers as the VGG16 net expects an RGB input
            tensorized = np.dstack([new_tile_list[i]] * 3)
 
            # create image from tensorized array           
            im = Image.fromarray(tensorized)
            
            # save to path specified in arguments
            im.save(tile_file_name)
            
            print("Tile with name {} written to disk".format(tile_file_name))

        counter += len(new_tile_list)

        print("So far {} files written".format(counter))       

    return 


def Main():

    write_all_MS_data_to_jpg()
    write_all_CLaMM_data_to_jpg()
        
    return

if __name__ == '__main__':
    Main()
