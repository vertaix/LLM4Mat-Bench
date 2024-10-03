#!/usr/bin/env bash

DATA_PATH='data/' # where LLM4Mat_Bench is saved
RESULTS_PATH='results/' # where to save the results
CHECKPOINTS_PATH='checkpoints/' # where to save model weights 
MODEL_NAME='llmprop' # or 'matbert'
DATASET_NAME='mp' # any dataset name in LLM4Mat_Bench
INPUT_TYPE='formula' # other values: 'cif_structure' and 'description'
PROPERTY_NAME='band_gap' # any property name in $DATASET_NAME. Please check the property names associated with each dataset first
MAX_LEN=256 # for testing purposes only, the default value is 888 while 2000 has shown to give the best performance
EPOCHS=5 #for testing purposes only, the default value is 200

python code/llmprop_and_matbert/train.py \
--data_path $DATA_PATH \
--results_path $RESULTS_PATH \
--checkpoints_path $CHECKPOINTS_PATH \
--model_name $MODEL_NAME \
--dataset_name $DATASET_NAME \
--input_type $INPUT_TYPE \
--property_name $PROPERTY_NAME \
--max_len $MAX_LEN \
--epochs $EPOCHS