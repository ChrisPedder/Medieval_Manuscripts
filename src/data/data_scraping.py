#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 11:07:05 2018

@author: chrispedder
"""

# Scientific computing libraries
import random as random

# File IO libraries
import os
from pathlib import Path

# Import timing libraries for exponential backoff when scraping
import time

# Import URL handling libraries
from bs4 import BeautifulSoup
import urllib3
from urllib.request import urlretrieve
import requests as requests

# import argparse to handle user input
import argparse

# use this image scraper from the location where you want to save images
# Used to retrieve jpg image files from http://image.ox.ac.uk/images/corpus/**
# Uses random exponential backoff to prevent server overload

def make_soup(url):
    http = urllib3.PoolManager()
    response = http.request('GET', url)
    return BeautifulSoup(response.data,"lxml")

def generate_image_links_list(url, file_extension):
    # generate beautiful soup data dump
    soupy = make_soup(url)
    # extract the entries that correspond to the file extension we want
    link_list = soupy.find_all('a')
    links = []
    for link in link_list:
        lnk = str(link).split('href=')[1]
        lnk2 = lnk.split('>')[0]
        links.append(lnk2)

    root_path = url[:21]
    final_list = []
    for elt in links:
        if elt[-4:] == 'jpg"':
            final_list.append(root_path + elt.strip('"'))

    return final_list

def download_images_to_directory(url_list, directory_extension = 'data/raw', 
                                 n=1):
    
    # define root path as location where python script is saved
    root_path = Path.cwd()
    
    # define the target directory path as root path plus extension
    target_directory_path = str(root_path.parent.parent) + '/' + directory_extension + '/'
    print(target_directory_path)
    
    #check chosen directory exists, if not create it
    if not os.path.exists(target_directory_path):
        os.mkdir(target_directory_path)
    
    # get total number of images
    total_images = len(url_list)
    print(str(total_images) + " images found.")
    print('Downloading images to directory {}.'.format(target_directory_path))
    
    #compile our unicode list of image links
    for number, image_link in enumerate(url_list):
        filename = image_link.split('/')[-1]
        time.sleep(min(64, (2 ** n)) + (random.randint(0, 1000) / 1000.0))
        img_data = requests.get(image_link).content
        with open(target_directory_path + filename, 'wb') as handler:
            handler.write(img_data)
        print('File number {} of {}, named {} downloaded'.format(number + 1, 
              str(total_images), filename))
    return 

def Main():
    # Define url to download images from
    parser = argparse.ArgumentParser()
    parser.add_argument('url', help='The URL from which you want to\
                        download documents', type=str)
    parser.add_argument('file_extension', help='The type of file you want to\
                        download from the specified URL', type=str)


    args = parser.parse_args()

    url = args.url
    file_extension = args.file_extension

    list_of_links = generate_image_links_list(url, file_extension)
    
    download_images_to_directory(list_of_links)    
    
    return

if __name__ == '__main__':
    Main()


