#!/usr/bin/env bash

cd ../../../
PYTHONPATH=. python tools/data/build_file_list.py household data/household/rawframes/ --num-split 1 --level 2 --subset train --format rawframes --shuffle
PYTHONPATH=. python tools/data/build_file_list.py household data/household/rawframes/ --num-split 1 --level 2 --subset val --format rawframes --shuffle
PYTHONPATH=. python tools/data/build_file_list.py household data/household/rawframes/ --num-split 1 --level 2 --subset test --format rawframes --shuffle

echo "Filelist for rawframes generated."

cd tools/data/household/
