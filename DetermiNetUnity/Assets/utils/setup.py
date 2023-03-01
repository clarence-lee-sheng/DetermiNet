import json 
import os
import shutil
import numpy as np 
import pandas as pd
import inflect

# This file will do the following: 
# 1. clear all files in the Streaming Assets dataset directory 
# 2. Create two directories in dataset directory, images and segmentations, and within the two directories, create one folder for each determiner
# 3. initialize pluralized text 


##############################################################################
############### Clear and reinitialised new dataset directory ################
##############################################################################

dir_path = os.path.dirname(os.path.realpath(__file__))
determiners = json.load(open(os.path.join(dir_path, '../config/determiners.json')))
dataset_dir = os.path.join(dir_path, "../StreamingAssets/dataset")
images_dir = "images"
segmentations_dir = "segmentations"

if os.path.exists(dataset_dir):
    shutil.rmtree(dataset_dir)
os.makedirs(dataset_dir)
os.makedirs(os.path.join(dataset_dir, images_dir))
os.makedirs(os.path.join(dataset_dir, segmentations_dir))
os.makedirs(os.path.join(dataset_dir, "annotations"))
# os.makedirs(os.path.join(dataset_dir, segmentations_dir, "annotations"))

for determiner in determiners:
    os.makedirs(os.path.join(dataset_dir, images_dir, determiner))
    os.makedirs(os.path.join(dataset_dir, segmentations_dir, determiner))

##############################################################################
####################### Generate Pluraliized JSON ############################
##############################################################################


cat_filename = os.path.join(dir_path, "../Scripts/config/categories.json")
plural_filename = os.path.join(dir_path, "../Scripts/config/plural.json")
p = inflect.engine()

categories = json.load(open(cat_filename))
plural = {}

for item in categories.keys(): 
    if categories[item]["supercategory"] == "countable": 
        plural_item = p.plural(item)
        plural[item] = plural_item 
    else: 
        plural[item] = item

json.dump(plural, open(plural_filename, 'w'))