import json 
import os
import shutil

# This file will do the following: 
# 1. clear all files in the Streaming Assets dataset directory 
# 2. Create two directories in dataset directory, images and segmentations, and within the two directories, create one folder for each determiner
# 3. initialize pluralized text 


##############################################################################
############### Clear and reinitialised new dataset directory ################
##############################################################################

determiners = json.load(open('../config/determiners.json'))
dataset_dir = "../StreamingAssets/dataset"
images_dir = "images"
segmentations_dir = "segmentations"

if os.path.exists(dataset_dir):
    shutil.rmtree(dataset_dir)
os.makedirs(dataset_dir)
os.makedirs(os.path.join(dataset_dir, images_dir))
os.makedirs(os.path.join(dataset_dir, segmentations_dir))
os.makedirs(os.path.join(dataset_dir, images_dir, "annotations"))
os.makedirs(os.path.join(dataset_dir, segmentations_dir, "annotations"))

for determiner in determiners:
    os.makedirs(os.path.join(dataset_dir, images_dir, determiner))
    os.makedirs(os.path.join(dataset_dir, segmentations_dir, determiner))

##############################################################################
####################### Generate Pluraliized JSON ############################
##############################################################################

# categories = json.load(open('../config/categories.json'))