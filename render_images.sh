#!/bin/bash

# Set directory containing object folders
shapenet_dir="/hdd2/obj_models/train"

# Set output directory
output_dir="/hdd2/nocs_category_level_v3"

num_threads=8

# Iterate through all folders in shapenet_dir and render images in parallel
ls -d "$shapenet_dir"/*/ | xargs -n 1 basename | parallel -j "$num_threads" python plot_nocs_pyrender.py {} "$shapenet_dir" "$output_dir"