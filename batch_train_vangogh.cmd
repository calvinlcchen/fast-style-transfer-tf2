rem ! /bin/bash

rem  NUM_FILTERS=8

rem starrynight - NHWC
python style.py --data-format NHWC --num-base-channels 8 --checkpoint-dir ckpts --test-dir example\output --style example\style\vangogh\starrynight-300-255.jpg --test example\content\sun-wallpaper-1280x720.jpg --batch-size 1 --content-weight 7e0 --style-weight 1e3 --learning-rate 1e-2 --checkpoint-iterations 10 --train-path ../data/train2014 --vgg-path ..\data\imagenet-vgg-verydeep-19.mat


