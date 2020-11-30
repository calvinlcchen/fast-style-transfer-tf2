rem ! /bin/bash

rem  NUM_FILTERS=8

rem starrynight - NHWC
python style.py --data-format NHWC --num-base-channels 32 --checkpoint-dir ckpts --test-dir example\output --style example\style\china\chun_sian_chi_ju.jpg --test example\content\sun-wallpaper-1280x720.jpg --batch-size 1 --content-weight 7 --style-weight 70  --checkpoint-iterations 10000 --train-path ../data/train2014 --vgg-path ..\data\imagenet-vgg-verydeep-19.mat

python style.py --data-format NHWC --num-base-channels 32 --checkpoint-dir ckpts --test-dir example\output --style example\style\china\chang_da_chian.jpg --test example\content\sun-wallpaper-1280x720.jpg --batch-size 1 --content-weight 7 --style-weight 70  --checkpoint-iterations 10000 --train-path ../data/train2014 --vgg-path ..\data\imagenet-vgg-verydeep-19.mat

rem python style.py --data-format NHWC --num-base-channels 32 --checkpoint-dir ckpts --test-dir example\output --style example\style\china\chang_da_chian.jpg --test example\content\sun-wallpaper-1280x720.jpg --batch-size 1 --content-weight 7 --style-weight 60  --checkpoint-iterations 10000 --train-path ../data/train2014 --vgg-path ..\data\imagenet-vgg-verydeep-19.mat

python style.py --data-format NHWC --num-base-channels 32 --checkpoint-dir ckpts --test-dir example\output --style example\style\china\un_dau_san_sui.jpeg --test example\content\sun-wallpaper-1280x720.jpg --batch-size 1 --content-weight 7 --style-weight 50  --checkpoint-iterations 10000 --train-path ../data/train2014 --vgg-path ..\data\imagenet-vgg-verydeep-19.mat

python style.py --data-format NHWC --num-base-channels 32 --checkpoint-dir ckpts --test-dir example\output --style example\style\china\un_dau_san_sui.jpeg --test example\content\sun-wallpaper-1280x720.jpg --batch-size 1 --content-weight 7 --style-weight 60  --checkpoint-iterations 10000 --train-path ../data/train2014 --vgg-path ..\data\imagenet-vgg-verydeep-19.mat

python style.py --data-format NHWC --num-base-channels 32 --checkpoint-dir ckpts --test-dir example\output --style example\style\china\un_dau_san_sui.jpeg --test example\content\sun-wallpaper-1280x720.jpg --batch-size 1 --content-weight 7 --style-weight 70  --checkpoint-iterations 10000 --train-path ../data/train2014 --vgg-path ..\data\imagenet-vgg-verydeep-19.mat

