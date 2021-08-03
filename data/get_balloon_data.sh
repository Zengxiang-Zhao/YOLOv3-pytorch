#!/bin/bash

# download balloon dataset
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
unzip -q balloon_dataset.zip

# create labels folders
mkdir balloon/train-labels
mkdir balloon/val-labels

python process_balloon_data.py --output balloon/train-labels --img_dir balloon/train/
python process_balloon_data.py --output balloon/val-labels --img_dir balloon/val/

# create names for test.py
echo balloon > balloon.names