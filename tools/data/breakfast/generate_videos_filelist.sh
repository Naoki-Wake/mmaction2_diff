#!/usr/bin/env bash

cd ../../../
PYTHONPATH=. python tools/data/build_file_list.py breakfast data/breakfast/videos/ --num-split 1 --level -1 --subset train --format videos --shuffle --out-root-path data/
PYTHONPATH=. python tools/data/build_file_list.py breakfast data/breakfast/videos/ --num-split 1 --level -1 --subset val --format videos --shuffle --out-root-path data/
PYTHONPATH=. python tools/data/build_file_list.py breakfast data/breakfast/videos/ --num-split 1 --level -1 --subset test --format videos --shuffle --out-root-path data/
echo "Filelist for videos generated."

cd tools/data/breakfast/
