#!/bin/bash

# Set directory containing object folders
shapenet_dir="/hdd2/real_camera_dataset/obj_models/train_selected"

# Set output directory
output_dir="/hdd2/nocs_category_level_symmetries"

num_threads=8

# Iterate through all folders in shapenet_dir and render images in parallel
ls -d "$shapenet_dir"/*/ | xargs -n 1 basename | parallel --line-buffer -j "$num_threads" python render_nocs_dataset_with_symmetries.py {} "$shapenet_dir" "$output_dir"