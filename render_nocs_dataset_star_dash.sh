#!/bin/bash

for x in {0..5}; do
python render_nocs_dataset_star_dash.py $x /home/domin/Documents/Datasets/nocs/train_selected /home/domin/Documents/Datasets/nocs/xyz_data
done