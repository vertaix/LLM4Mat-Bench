# LLM4Mat-Bench
LLM4Mat-Bench is the largest benchmark to date for evaluating the performance of large language models (LLMs) for materials property prediction.

<p align="center" width="100%">
    <img src="figures/llm4mat-bench_stats.png" alt="image" width="100%" height="auto">
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

## Leaderboard
<div style="overflow-x:auto;">
<table>
  <thead>
    <tr>
      <th rowspan="3">Input</th>
      <th rowspan="3">Model</th>
      <th colspan="2"> MP</th>
      <th>JARVIS-DFT</th>
      <th>GNoME</th>
      <th>hMOF</th>
      <th>Cantor HEA</th>
      <th>JARVIS-QETB</th>
      <th>OQMD</th>
      <th>QMOF</th>
      <th colspan="2">SNUMAT</th>
      <th>OMDB</th>
    </tr>
    <tr>
      <th> Regression </th>
      <th> Classification </th>
      <th> Regression </th>
      <th> Regression </th>
      <th> Regression </th>
      <th> Regression </th>
      <th> Regression </th>
      <th> Regression </th>
      <th> Regression </th>
      <th> Classification </th>
      <th> Regression </th>
      <th> Regression </th>
    </tr>
    <tr>
      <th> 8 tasks </th>
      <th> 2 tasks </th>
      <th> 20 tasks </th>
      <th> 6 tasks </th>
      <th> 7 tasks </th>
      <th> 6 tasks </th>
      <th> 4 tasks </th>
      <th> 2 tasks </th>
      <th> 4 tasks </th>
      <th> 4 tasks </th>
      <th> 3 tasks </th>
      <th> 1 task </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>CIF</td>
      <td>CGCNN (baseline)</td>
      <td>5.319</td>
      <td>0.846</td>
      <td>7.048</td>
      <td>19.478</td>
      <td>2.257</td>
      <td>17.780</td>
      <td>61.729</td>
      <td>14.496</td>
      <td>3.076</td>
      <td>1.973</td>
      <td>0.722</td>
      <td>2.751</td>
    </tr>
    <tr>
      <td rowspan="4">Comp.</td>
      <td>Llama 2-7b-chat:0S</td>
      <td>0.389</td>
      <td>0.491</td>
      <td>Inval.</td>
      <td>0.164</td>
      <td>0.174</td>
      <td>0.034</td>
      <td>0.188</td>
      <td>0.105</td>
      <td>0.303</td>
      <td>0.940</td>
      <td>Inval.</td>
      <td>0.885</td>
    </tr>
    <tr>
      <td>Llama 2-7b-chat:5S</td>
      <td>0.627</td>
      <td>0.507</td>
      <td>0.704</td>
      <td>0.499</td>
      <td>0.655</td>
      <td>0.867</td>
      <td>1.047</td>
      <td>1.160</td>
      <td>0.932</td>
      <td>1.157</td>
      <td>0.466</td>
      <td>1.009</td>
    </tr>
    <tr>
      <td>MatBERT-109M</td>
      <td><b>5.317</b></td>
      <td>0.722</td>
      <td><b>4.103</b></td>
      <td>12.834</td>
      <td>1.430</td>
      <td>6.769</td>
      <td>11.952</td>
      <td>5.772</td>
      <td><b>2.049</b></td>
      <td><b>1.828</b></td>
      <td>0.712</td>
      <td><b>1.554</b></td>
    </tr>
    <tr>
      <td>LLM-Prop-35M</td>
      <td>4.394</td>
      <td>0.691</td>
      <td>2.912</td>
      <td><b>15.599</b></td>
      <td><b>1.479</b></td>
      <td><b>8.400</b></td>
      <td><b>59.443</b></td>
      <td><b>6.020</b></td>
      <td>1.958</td>
      <td>1.509</td>
      <td>0.719</td>
      <td>1.507</td>
    </tr>
    <tr>
      <td rowspan="4">CIF</td>
      <td>Llama 2-7b-chat:0S</td>
      <td>0.392</td>
      <td>0.501</td>
      <td>0.216</td>
      <td>6.746</td>
      <td>0.214</td>
      <td>0.022</td>
      <td>0.278</td>
      <td>0.028</td>
      <td>0.119</td>
      <td>0.682</td>
      <td>0.489</td>
      <td>0.159</td>
    </tr>
    <tr>
      <td>Llama 2-7b-chat:5S</td>
      <td>Inval.</td>
      <td>0.502</td>
      <td>Inval.</td>
      <td>Inval.</td>
      <td>Inval.</td>
      <td>Inval.</td>
      <td>1.152</td>
      <td>1.391</td>
      <td>Inval.</td>
      <td>Inval.</td>
      <td>0.474</td>
      <td>0.930</td>
    </tr>
    <tr>
      <td>MatBERT-109M</td>
      <td>7.452</td>
      <td>0.750</td>
      <td>6.211</td>
      <td>14.227</td>
      <td>1.514</td>
      <td>9.958</td>
      <td>47.687</td>
      <td>10.521</td>
      <td>3.024</td>
      <td><b>2.131</b></td>
      <td>0.717</td>
      <td><b>1.777</b></td>
    </tr>
    <tr>
      <td>LLM-Prop-35M</td>
      <td><b>8.554</b></td>
      <td>0.738</td>
      <td><b>6.756</b></td>
      <td><b>16.032</b></td>
      <td><b>1.623</b></td>
      <td><b>15.728</b></td>
      <td><b>97.919</b></td>
      <td><b>11.041</b></td>
      <td><b>3.076</b></td>
      <td>1.829</td>
      <td>0.660</td>
      <td><b>1.777</b></td>
    </tr>
    <tr>
      <td rowspan="4">Descr.</td>
      <td>Llama 2-7b-chat:0S</td>
      <td>0.437</td>
      <td>0.500</td>
      <td>0.247</td>
      <td>0.336</td>
      <td>0.193</td>
      <td>0.069</td>
      <td>0.264</td>
      <td>0.106</td>
      <td>0.152</td>
      <td>0.883</td>
      <td>Inval.</td>
      <td>0.155</td>
    </tr>
    <tr>
      <td>Llama 2-7b-chat:5S</td>
      <td>0.635</td>
      <td>0.502</td>
      <td>0.703</td>
      <td>0.470</td>
      <td>0.653</td>
      <td>0.820</td>
      <td>0.980</td>
      <td>1.230</td>
      <td>0.946</td>
      <td>1.040</td>
      <td>0.568</td>
      <td>1.001</td>
    </tr>
    <tr>
      <td>MatBERT-109M</td>
      <td>7.651</td>
      <td>0.735</td>
      <td>6.083</td>
      <td>15.558</td>
      <td>1.558</td>
      <td>9.976</td>
      <td>46.586</td>
      <td><b>11.027</b></td>
      <td><b>3.055</b></td>
      <td><b>2.152</b></td>
      <td>0.730</td>
      <td><b>1.847</b></td>
    </tr>
    <tr>
      <td>LLM-Prop-35M</td>
      <td><b>9.116</b></td>
      <td>0.742</td>
      <td><b>7.204</b></td>
      <td><b>16.224</b></td>
      <td><b>1.706</b></td>
      <td><b>15.926</b></td>
      <td>93.001</td>
      <td>9.995</td>
      <td>3.016</td>
      <td>1.950</td>
      <td>0.735</td>
      <td>1.656</td>
    </tr>
  </tbody>
</table>
</div>


