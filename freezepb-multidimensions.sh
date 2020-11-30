#! /bin/bash

# $1 : data-format
# $2 : checkpoint path
# $3 : num-base-channels 
# $4 : model filename name


python3.6 evaluate.py --data-format $1 --num-base-channels $3 --checkpoint $2 --in-path example/content/sun-wallpaper-640x480.jpg --out-path example/result --allow-different-dimensions
python3.6 -m tensorflow.python.tools.freeze_graph --input_graph=$2/graph.pbtxt --input_checkpoint=$2/saver --output_graph=$2/$4-$1-nbc$3-640x480.pb --output_node_names="output"
tflite_convert --output_file=$2/$4-$1-nbc$3-640x480.tflite --graph_def_file=$2/$4-$1-nbc$3-640x480.pb --inference_type=FLOAT --input_arrays=img_placeholder --output_arrays=output

python3.6 evaluate.py --data-format $1 --num-base-channels $3 --checkpoint $2 --in-path example/content/sun-wallpaper-1280x720.jpg --out-path example/result --allow-different-dimensions
python3.6 -m tensorflow.python.tools.freeze_graph --input_graph=$2/graph.pbtxt --input_checkpoint=$2/saver --output_graph=$2/$4-$1-nbc$3-1280x720.pb --output_node_names="output"
tflite_convert --output_file=$2/$4-$1-nbc$3-1280x720.tflite --graph_def_file=$2/$4-$1-nbc$3-1280x720.pb --inference_type=FLOAT --input_arrays=img_placeholder --output_arrays=output

python3.6 evaluate.py --data-format $1 --num-base-channels $3 --checkpoint $2 --in-path example/content/sun-wallpaper-1920x1080.jpg --out-path example/result --allow-different-dimensions
python3.6 -m tensorflow.python.tools.freeze_graph --input_graph=$2/graph.pbtxt --input_checkpoint=$2/saver --output_graph=$2/$4-$1-nbc$3-1920x1080.pb --output_node_names="output"
tflite_convert --output_file=$2/$4-$1-nbc$3-1920x1080.tflite --graph_def_file=$2/$4-$1-nbc$3-1920x1080.pb --inference_type=FLOAT --input_arrays=img_placeholder --output_arrays=output
