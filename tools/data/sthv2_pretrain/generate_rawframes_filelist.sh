#!/usr/bin/env bash

cd ../../../
PYTHONPATH=. python tools/data/build_file_list.py sthv2_pretrain data/sthv2/rawframes/ --num-split 1 --level 1 --subset train --format rawframes --shuffle
PYTHONPATH=. python tools/data/build_file_list.py sthv2_pretrain data/sthv2/rawframes/ --num-split 1 --level 1 --subset val --format rawframes --shuffle


echo "Filelist for rawframes generated."

cd tools/data/household/
