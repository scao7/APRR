# Preprocess video data to to the HumanNeRF needed data
# input path videos/trian output path humannerf/processed
# With AMR and without AMR

# subejct
#     ├── images
#     │   └── ${item_id}.png
#     ├── masks
#     │   └── ${item_id}.png
#     └── metadata.json

import cv2
import torch
import os 
import numpy as np 
import romp
from process_util import *
import json

training_video = "/data/Shengting/paperII/videos/training/IMG_1859_train.mp4"
subejct_path = "/data/Shengting/paperII/processed_data/IMG_1859_train_L"
subejct_images_path = os.path.join(subejct_path,"images")
subject_masks_path = os.path.join(subejct_path,"masks")
subejct_metadata_path = subejct_path

# # # mask 
frame_interval = 10  # Save every 10th frame, adjust as needed
video_to_selected(training_video, subejct_images_path, frame_interval,downsample_rate=1,match_threshold=0.4)

# Create the output folder if it doesn't exist
if not os.path.exists(subject_masks_path):
    os.makedirs(subject_masks_path)

for i in os.listdir(subejct_images_path):
    mask  = segment_human(os.path.join(subejct_images_path,i))
    save_mask_as_png(mask,os.path.join(subject_masks_path,i))


# setup rompestimation and generate metadata.json
print(os.listdir(subejct_images_path)[0])
first_image_path = os.path.join(subejct_images_path,
    os.listdir(subejct_images_path)[0])
imsize = cv2.imread(first_image_path).shape[:2]
imsize = imsize[::-1]
print(imsize)
print(type(imsize))
fov = 60
H = max(imsize)
focal_length=H/2. * 1./np.tan(np.radians(fov/2))
# metadata = rompEstimation(subejct_images_path,focal_length,imsize)
metadata = rompSelection(subejct_images_path,focal_length,imsize)
with open(os.path.join(subejct_path,"metadata.json"), "w") as outfile:
        json.dump(metadata, outfile)

# run the prepare_wild script

# Change the current working directory
os.chdir("/data/Shengting/paperII/humannerf/tools/prepare_wild")
# Verify the change by printing the current working directory
print(f"Current working directory: {os.getcwd()}")
os.system("python prepare_dataset.py")
os.chdir("/data/Shengting/paperII")
# Verify the change by printing the current working directory
print(f"Current working directory: {os.getcwd()}")