#!/usr/bin/env bash

DATA_PATH='data/' # where LLM4Mat_Bench is saved
RESULTS_PATH='results/' # where to save the results
DATASET_NAME='mp' # any dataset name in LLM4Mat_Bench
INPUT_TYPE='formula' # other values: 'cif_structure' and 'description'
PROPERTY_NAME='band_gap' # any property name in $DATASET_NAME. Please check the property names associated with each dataset first
PROMPT_TYPE='zero_shot' # 'few_shot' can also be used here which let llama see five examples before it generates the answer
MAX_LEN=800 # max_len and batch_size can be modified according to the available resources
BATCH_SIZE=8
MIN_SAMPLES=2 # minimum number of valid outputs from llama (the default number is 10)

python code/llama/evaluate.py \
--data_path $DATA_PATH \
--results_path $RESULTS_PATH \
--dataset_name $DATASET_NAME \
--input_type $INPUT_TYPE \
--property_name $PROPERTY_NAME \
--prompt_type $PROMPT_TYPE \
--max_len $MAX_LEN \
--batch_size $BATCH_SIZE \
--min_samples $MIN_SAMPLES