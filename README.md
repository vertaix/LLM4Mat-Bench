# LLM4Mat-Bench
LLM4Mat-Bench is the largest benchmark to date for evaluating the performance of large language models (LLMs) for materials property prediction.

<p align="center" width="100%">
    <img src="figures/llm4mat-bench_stats.png" alt="image" width="50%" height="auto">
    <br>
    <em>LLM4Mat-Bench Statistics. *https://www.snumat.com/apis</em>
</p>

## How to use
### Generating the property values with LLaMA2-7b-chat model
Add the following scripts to [llama_inference.sh](scripts/llama_inference.sh)
```bash
#!/usr/bin/env bash

DATA_PATH='data/' # where LLM4Mat_Bench is saved
RESULTS_PATH='results/' # where to save the results
DATASET_NAME='mp' # any dataset name in LLM4Mat_Bench
INPUT_TYPE='formula' # other values: 'cif_structure' and 'description'
PROPERTY_NAME='band_gap' # any property name in $DATASET_NAME. Please check the property names associated with each dataset first
PROMPT_TYPE='zero_shot' # 'few_shot' can also be used here which let llama see five examples before it generates the answer
MAX_LEN=800 # max_len and batch_size can be modified according to the available resources
BATCH_SIZE=8

python code/llama/llama_inference.py \
--data_path $DATA_PATH \
--results_path $RESULTS_PATH \
--dataset_name $DATASET_NAME \
--input_type $INPUT_TYPE \
--property_name $PROPERTY_NAME \
--prompt_type $PROMPT_TYPE \
--max_len $MAX_LEN \
--batch_size $BATCH_SIZE
```
Then run ```bash scripts/llama_inference.sh```

### Evaluating the LLaMA results
After running ```bash scripts/llama_inference.sh```, add the following scripts to [llama_evaluate.sh](scripts/llama_inference.sh)
```bash
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
```
Then run ```bash scripts/llama_evaluate.sh```

## Data Availability
The data collected for the LLM4Mat-Bench can be found [here](https://drive.google.com/drive/folders/1HpGhuNHG4EQCQMZaKPwEQNH9stJKw-ht). Each dataset includes a fixed train/validation/test split for reproducibility and fair model comparison. The data **LICENSE** belongs to the original creators of each dataset/database.

## TODOs
- Adding the link to the paper
- Adding the detailed guidelines on how to run the models we evaluated on LLM4Mat-Bench
- Adding how to cite LLM4Mat-Bench
- Adding the leaderboard (Optional)