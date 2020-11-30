rem %1 : data-format
rem %2 : checkpoint path
rem %3 : num-base-channels 
rem %4 : model filename name
rem %5 : resolution  ex, 640x480

python evaluate.py --data-format %1 --num-base-channels %3 --checkpoint %2 --in-path example/content/sun-wallpaper-%5.jpg --out-path example/result --allow-different-dimensions
python -m tensorflow.python.tools.freeze_graph --input_graph=%2/graph.pbtxt --input_checkpoint=%2/saver --output_graph=%2/%4-%1-nbc%3-%5.pb --output_node_names="output"

rem python C:\practice\AIToolkit\tensorflow\tensorflow\python\tools\freeze_graph.py --input_graph=%2/graph.pbtxt --input_checkpoint=%2/saver --output_graph=%2/%4-%1-nbc%3-%5.pb --output_node_names="output"
