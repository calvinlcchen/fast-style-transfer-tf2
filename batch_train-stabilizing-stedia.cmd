rem ! /bin/bash

rem  NUM_FILTERS=4



python style.py --data-format NCHW --num-base-channels 8 --checkpoint-dir ckpts --test-dir example/output  --style example/style/Stedia/stedia-1.jpg --test example\content\sun-wallpaper-1280x720.jpg --batch-size 1 --content-weight 7.5 --style-weight 100 --checkpoint-iterations 10000 --train-path ..\data\train2014 --vgg-path ..\data\imagenet-vgg-verydeep-19.mat 
python style.py --data-format NCHW --num-base-channels 8 --checkpoint-dir ckpts --test-dir example/output  --style example/style/Stedia/stedia-3.jpg --test example\content\sun-wallpaper-1280x720.jpg --batch-size 1 --content-weight 7.5 --style-weight 100 --checkpoint-iterations 10000 --train-path ..\data\train2014 --vgg-path ..\data\imagenet-vgg-verydeep-19.mat 
python style.py --data-format NCHW --num-base-channels 8 --checkpoint-dir ckpts --test-dir example/output  --style example/style/Stedia/stedia-4.jpg --test example\content\sun-wallpaper-1280x720.jpg --batch-size 1 --content-weight 7.5 --style-weight 100 --checkpoint-iterations 10000 --train-path ..\data\train2014 --vgg-path ..\data\imagenet-vgg-verydeep-19.mat 
python style.py --data-format NCHW --num-base-channels 8 --checkpoint-dir ckpts --test-dir example/output  --style example/style/Stedia/Starry_Night-321x255.jpg --test example\content\sun-wallpaper-1280x720.jpg --batch-size 1 --content-weight 7.5 --style-weight 100 --checkpoint-iterations 10000 --train-path ..\data\train2014 --vgg-path ..\data\imagenet-vgg-verydeep-19.mat 

