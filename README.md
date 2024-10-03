# LLM4Mat-Bench
LLM4Mat-Bench is the largest benchmark to date for evaluating the performance of large language models (LLMs) for materials property prediction.

<p align="center" width="100%">
    <img src="figures/llm4mat-bench_stats.png" alt="image" width="50%" height="auto">
    <br>
    <em>LLM4Mat-Bench Statistics. *https://www.snumat.com/apis</em>
</p>

## How to use
### Installation
```
git clone https://github.com/vertaix/LLM4Mat-Bench.git
cd LLM4Mat-Bench
conda create -n <environment_name> requirement.txt
conda activate <environment_name>
```
### Get the data
- Download the LLM4Mat-Bench data from [this link](https://drive.google.com/drive/folders/1HpGhuNHG4EQCQMZaKPwEQNH9stJKw-ht). Each dataset includes a fixed train/validation/test split for reproducibility and fair model comparison. 
- Save the data into [data folder](data/) where LLM4Mat-Bench is the parent directory.

### Get the checkpoints
- Download the LLM-Prop and MatBERT checkpoints from [this link]().
- Save the checkpoints folder into LLM4Mat-Bench directory.

### Evaluating the trained LLM-Prop and MatBERT
Add any modification to the following scripts to [evaluate.sh](scripts/evaluate.sh)
```bash
#!/usr/bin/env bash

DATA_PATH='data/' # where LLM4Mat_Bench data is saved
RESULTS_PATH='results/' # where to save the results
CHECKPOINTS_PATH='checkpoints/' # where model weights were saved
MODEL_NAME='llmprop' # or 'matbert'
DATASET_NAME='mp' # any dataset name in LLM4Mat_Bench
INPUT_TYPE='formula' # other values: 'cif_structure' and 'description'
PROPERTY_NAME='band_gap' # any property name in $DATASET_NAME. Please check the property names associated with each dataset first

python code/llmprop_and_matbert/evaluate.py \
--data_path $DATA_PATH \
--results_path $RESULTS_PATH \
--checkpoints_path $CHECKPOINTS_PATH \
--model_name $MODEL_NAME \
--dataset_name $DATASET_NAME \
--input_type $INPUT_TYPE \
--property_name $PROPERTY_NAME
``` 
Then run 
```bash
 bash scripts/evaluate.sh
 ```

### Training LLM-Prop and MatBERT from scratch
Add any modification to the following scripts to [train.sh](scripts/train.sh)
```bash
#!/usr/bin/env bash

DATA_PATH='data/' # where LLM4Mat_Bench data is saved
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
```
Then run 
```bash
bash scripts/train.sh
```

### Generating the property values with LLaMA2-7b-chat model
Add any modification to the following scripts to [llama_inference.sh](scripts/llama_inference.sh)
```bash
#!/usr/bin/env bash

DATA_PATH='data/' # where LLM4Mat_Bench data is saved
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
Then run 
```bash
bash scripts/llama_inference.sh
```

### Evaluating the LLaMA results
After running ```bash scripts/llama_inference.sh```, add any modification to the following scripts to [llama_evaluate.sh](scripts/llama_evaluate.sh)
```bash
#!/usr/bin/env bash

DATA_PATH='data/' # where LLM4Mat_Bench data is saved
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
Then run 
```bash 
bash scripts/llama_evaluate.sh
```

## Data LICENSE
The data **LICENSE** belongs to the original creators of each dataset/database.
