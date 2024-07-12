#!/bin/bash

for x in {0..5}; do
python render_nocs_dataset_star_dash.py $x /home/domin/Documents/Datasets/nocs/real_test /home/domin/Documents/Datasets/nocs/xyz_data_test
done