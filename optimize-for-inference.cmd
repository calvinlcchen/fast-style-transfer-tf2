rem %1 : checkpoint path
rem %2 : model filename name

python -m tensorflow.python.tools.optimize_for_inference --input=%1/saver.data-00000-of-00001 --output_graph=%1/%2.optimize --input_names="img_placeholder" --output_names="output"
