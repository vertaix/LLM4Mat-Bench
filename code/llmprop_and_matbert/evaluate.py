import os
import re
import glob
import time
import datetime
import random
from datetime import timedelta

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim import SGD

import matplotlib.pyplot as plt

# add the progress bar
from tqdm import tqdm

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer, BertTokenizerFast, BertModel
from tokenizers.pre_tokenizers import Whitespace

pre_tokenizer = Whitespace()

# pre-defined functions
from model import Predictor
from utils import *
from dataset import *
from create_args_parser import *

# for metrics
from torchmetrics.classification import BinaryAUROC
from sklearn import metrics
from scipy.stats import pearsonr, spearmanr, kendalltau
from statistics import stdev

from datetime import timedelta

import subprocess

def evaluate(
    model, 
    mae_loss_function, 
    test_dataloader, 
    train_labels_mean, 
    train_labels_std, 
    property,
    device,
    task_name,
    normalizer="z_norm"
):
    test_start_time = time.time()

    model.eval()

    total_test_loss = 0
    predictions_list = []
    targets_list = []

    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            if preprocessing_strategy == 'xVal':
                batch_inputs, batch_masks, batch_labels, batch_x_num = tuple(b.to(device) for b in batch)
                _, predictions = model(batch_inputs, batch_masks, x_num=batch_x_num)
            else:
                batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
                _, predictions = model(batch_inputs, batch_masks)

            if task_name == "classification":
                predictions_denorm = predictions

            elif task_name == "regression":
                if normalizer == 'z_norm':
                    predictions_denorm = z_denormalize(predictions, train_labels_mean, train_labels_std)

                elif normalizer == 'mm_norm':
                    predictions_denorm = mm_denormalize(predictions, train_labels_min, train_labels_max)

                elif normalizer == 'ls_norm':
                    predictions_denorm = ls_denormalize(predictions)

                elif normalizer == 'no_norm':
                    predictions_denorm = predictions

        predictions_detached = predictions_denorm.detach().cpu().numpy()
        targets = batch_labels.detach().cpu().numpy()

        for i in range(len(predictions_detached)):
            predictions_list.append(predictions_detached[i][0])
            targets_list.append(targets[i])
        
    if task_name == "classification":
        test_performance = get_roc_score(predictions_list, targets_list)
        print(f"\n Test ROC score on predicting {property} = {test_performance}")

    elif task_name == "regression":
        predictions_tensor = torch.tensor(predictions_list)
        targets_tensor = torch.tensor(targets_list)
        test_performance = mae_loss_function(predictions_tensor.squeeze(), targets_tensor.squeeze())
        rmse = metrics.mean_squared_error(targets_list, predictions_list, squared=False)
        r2 = metrics.r2_score(targets_list, predictions_list)

        # correlations between targets_list and predictions_list
        pearson_r, pearson_p_value = pearsonr(targets_list, predictions_list)
        spearman_r, spearman_p_value = spearmanr(targets_list, predictions_list)
        kendall_r, kendall_p_value = kendalltau(targets_list, predictions_list)

        print(f"\n The test performance on predicting {property}:")
        print(f"MAE error = {test_performance}")
        print(f"RMSE error = {rmse}")
        print(f"R2 score = {r2}")
        print(f"Pearson_r = {pearson_r}", f"Pearson_p_value = {pearson_p_value}")
        print(f"Spearman_r = {spearman_r}", f"Spearman_p_value = {spearman_p_value}")
        print(f"Kendall_r = {kendall_r}", f"Kendall_p_value = {kendall_p_value}")

    average_test_loss = total_test_loss / len(test_dataloader)
    test_ending_time = time.time()
    testing_time = time_format(test_ending_time-test_start_time)
    print(f"Testing took {testing_time} \n")

    return predictions_list, test_performance

if __name__ == "__main__":
    # check if the GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'Number of available devices: {torch.cuda.device_count()}')
        print(f'Current device is: {torch.cuda.current_device()}')
        print("Evaluating on", torch.cuda.device_count(), "GPUs!")
        print('-'*50)
    else:
        print("No GPU available, please connect to the GPU first or continue to use CPU instead")
        print('-'*50)
        device = torch.device("cpu")
        
    # parse Arguments
    args = args_parser()
    config = vars(args)
    
    inference_batch_size = config.get('inference_bs')
    max_length = config.get('max_len')
    drop_rate = config.get('dr')
    preprocessing_strategy = config.get('preprocessing_strategy')
    tokenizer_name = config.get('tokenizer')
    pooling = config.get('pooling')
    normalizer_type = config.get('normalizer')
    property = config.get('property_name')
    data_path = config.get('data_path')
    input_type = config.get('input_type')
    dataset_name = config.get('dataset_name')
    model_name = config.get('model_name')
    checkpoints_path = config.get('checkpoints_path')
    results_path = config.get('results_path')
    tokenizers_path = config.get('tokenizers_path')
    
    if model_name == "matbert":
        pooling = None

    # checkpoints directory
    checkpoints_path = f"{checkpoints_path}/{dataset_name}/"
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    # training statistics directory
    results_path = f"{results_path}/{dataset_name}/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # prepare the data
    train_data = pd.read_csv(f"{data_path}/{dataset_name}/train.csv")
    valid_data = pd.read_csv(f"{data_path}/{dataset_name}/validation.csv")
    test_data = pd.read_csv(f"{data_path}/{dataset_name}/test.csv")
    
    if dataset_name == "mp":
        test_data = test_data.rename(columns={"formula_pretty":"formula"})
        valid_data = valid_data.rename(columns={"formula_pretty":"formula"})
        train_data = train_data.rename(columns={"formula_pretty":"formula"})
    elif dataset_name == "cantor_hea":
        test_data = test_data.rename(columns={"reduced_formula":"formula"})
        valid_data = valid_data.rename(columns={"reduced_formula":"formula"})
        train_data = train_data.rename(columns={"reduced_formula":"formula"})
    
    # drop duplicates in test data
    test_data = test_data.drop_duplicates(subset=['material_id']).reset_index(drop=True)
    
    # drop samples with nan input values
    train_data = train_data.dropna(subset=[input_type]).reset_index(drop=True)
    valid_data = valid_data.dropna(subset=[input_type]).reset_index(drop=True)
    test_data = test_data.dropna(subset=[input_type]).reset_index(drop=True)

    if dataset_name in ["gnome"]:
        # changing 'inf' values to 'NaN'
        train_data[property] = train_data[property][np.isfinite(train_data[property])]
        test_data[property] = test_data[property][np.isfinite(test_data[property])]

    # drop samples with nan property values
    train_data = train_data.dropna(subset=[property]).reset_index(drop=True)
    test_data = test_data.dropna(subset=[property]).reset_index(drop=True)
    
    print("\n number of test samples = ", len(test_data),'\n')

    # check property type to determine the task name (whether it is regression or classification)
    if train_data[property].dtype in ['bool', 'O']:
        task_name = 'classification'
        
        if property in ['Direct_or_indirect', 'Direct_or_indirect_HSE']:
            #converting Direct->1.0 and Indirect->0.0
            train_data = train_data.drop(train_data[train_data[property] == 'Null'].index).reset_index(drop=True)
            train_data.loc[train_data[property] == "Direct", property] = 1.0
            train_data.loc[train_data[property] == "Indirect", property] = 0.0
            train_data[property] = train_data[property].astype(float)

            test_data = test_data.drop(test_data[test_data[property] == 'Null'].index).reset_index(drop=True)
            test_data.loc[test_data[property] == "Direct", property] = 1.0
            test_data.loc[test_data[property] == "Indirect", property] = 0.0
            test_data[property] = test_data[property].astype(float)
        else:
            #converting True->1.0 and False->0.0
            train_data[property] = train_data[property].astype(float)
            test_data[property] = test_data[property].astype(float)  
    else:
        task_name = 'regression'
    
    train_labels_array = np.array(train_data[property])
    train_labels_mean = torch.mean(torch.tensor(train_labels_array))
    train_labels_std = torch.std(torch.tensor(train_labels_array))
    train_labels_min = torch.min(torch.tensor(train_labels_array))
    train_labels_max = torch.max(torch.tensor(train_labels_array))

    if preprocessing_strategy == "none":
        test_data = test_data

    elif preprocessing_strategy == "bond_lengths_replaced_with_num":
        test_data['description'] = test_data['description'].apply(replace_bond_lengths_with_num)
        print(train_data['description'][0])
        print('-'*50)
        print(test_data['description'][3])

    elif preprocessing_strategy == "bond_angles_replaced_with_ang":
        test_data['description'] = test_data['description'].apply(replace_bond_angles_with_ang) 
        print(train_data['description'][0])
        print('-'*50)
        print(test_data['description'][3])

    elif preprocessing_strategy == "no_stopwords":
        if input_type == "description":
            test_data[input_type] = test_data[input_type].apply(remove_mat_stopwords)

        print(train_data.head(1))
        print('-'*50)
        print(train_data[input_type][0])
        print('-'*50)
        print(test_data[input_type][0])

    elif preprocessing_strategy == "no_stopwords_and_lengths_and_angles_replaced":
        if input_type == "description":
            test_data['description'] = test_data['description'].apply(replace_bond_lengths_with_num)
            test_data['description'] = test_data['description'].apply(replace_bond_angles_with_ang)
            test_data['description'] = test_data['description'].apply(remove_mat_stopwords)
        print(train_data['description'][0])
        print('-'*50)
        print(test_data['description'][3])

    elif preprocessing_strategy == "no_stopwords_and_remove_bond_lengths_and_angles":
        if input_type == "description":
            test_data['description'] = test_data['description'].apply(remove_bond_lengths_and_angles)
            test_data['description'] = test_data['description'].apply(remove_mat_stopwords)
        print(train_data['description'][0])
        print('-'*50)
        print(test_data['description'][3])

    elif preprocessing_strategy == "xVal":
        test_data['list_of_numbers_in_input'] = test_data[input_type].apply(get_numbers_in_a_sentence)
        test_data[input_type] = test_data[input_type].apply(replace_numbers_with_num)

        if input_type == "description":
            test_data[input_type] = test_data[input_type].apply(remove_mat_stopwords)

        print(train_data.head(1))
        print('-'*50)
        print(train_data[input_type][0])
        print('-'*50)
        print(test_data[input_type][0])

    # define loss functions
    mae_loss_function = nn.L1Loss()
    bce_loss_function = nn.BCEWithLogitsLoss()

    freeze = False # a boolean variable to determine if we freeze the pre-trained T5 weights

    # define the tokenizer
    if tokenizer_name == 't5_tokenizer':
        tokenizer = AutoTokenizer.from_pretrained("t5-small") 

    elif tokenizer_name == 'llmprop_tokenizer' or model_name == 'llmprop':
        tokenizer = AutoTokenizer.from_pretrained(f"{tokenizers_path}/llmprop_tokenizer") 

    elif tokenizer_name == 'matbert_tokenizer' or model_name == 'matbert':
        tokenizer = BertTokenizerFast.from_pretrained(f"{tokenizers_path}/matbert-base-uncased", do_lower_case=True)

    # add defined special tokens to the tokenizer
    if pooling == 'cls':
        tokenizer.add_tokens(["[CLS]"])

    if preprocessing_strategy == "bond_lengths_replaced_with_num":
        tokenizer.add_tokens(["[NUM]"]) # special token to replace bond lengths
    
    elif preprocessing_strategy == "bond_angles_replaced_with_ang":
        tokenizer.add_tokens(["[ANG]"]) # special token to replace bond angles

    elif preprocessing_strategy == "no_stopwords_and_lengths_and_angles_replaced":
        tokenizer.add_tokens(["[NUM]"])
        tokenizer.add_tokens(["[ANG]"])

    elif preprocessing_strategy == "xVal":
        tokenizer.add_tokens(["[NUM]"]) 

    #get the length of the longest composition
    if input_type in ["reduced_formula", "formula_pretty", "formula"]:
        max_length = get_max_len(pd.concat([train_data, valid_data, test_data]), tokenizer, input_type)
        print('\nThe longest composition has', max_length, 'tokens\n')

    print('max length:', max_length)

    print('-'*50)
    print(f"test data = {len(test_data)} samples") 
    print('-'*50)
    print(f"testing on {get_sequence_len_stats(test_data, tokenizer, max_length, input_type)}% samples with whole sequence")
    print('-'*50)

    print("labels statistics on training set:")
    print("Mean:", train_labels_mean)
    print("Standard deviation:", train_labels_std)
    print("Max:", train_labels_max)
    print("Min:", train_labels_min)
    print("-"*50)
    
    print("======= Evaluating on test set ========")
    
    print('\nResults for', f"{model_name}_best_checkpoint_for_{property}_{task_name}_{input_type}_{preprocessing_strategy}.pt\n")
    
    # averaging the results over 5 runs
    predictions = []
    test_results = []
    seed = 42
    offset = 10
    
    for i in range(5):
        np.random.seed(seed + (i*offset))
        random.seed(seed + (i*offset))
        torch.manual_seed(seed + (i*offset))
        torch.cuda.manual_seed(seed + (i*offset))
        
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed + (i*offset))

        # define the model
        if model_name == "llmprop":
            base_model = T5EncoderModel.from_pretrained("google/t5-v1_1-small")
            base_model_output_size = 512
        elif model_name == "matbert":
            base_model = BertModel.from_pretrained(f"{tokenizer_path}/matbert-base-uncased")
            base_model_output_size = 768

        # freeze the pre-trained LM's parameters
        if freeze:
            for param in base_model.parameters():
                param.requires_grad = False

        # resizing the model input embeddings matrix to adapt to newly added tokens by the new tokenizer
        # this is to avoid the "RuntimeError: CUDA error: device-side assert triggered" error
        base_model.resize_token_embeddings(len(tokenizer))
        
        best_model_path = f"{checkpoints_path}/{model_name}_best_checkpoint_for_{property}_{task_name}_{input_type}_{preprocessing_strategy}.pt"
        best_model = Predictor(base_model, base_model_output_size, drop_rate=drop_rate, pooling=pooling, model_name=model_name)

        device_ids = [d for d in range(torch.cuda.device_count())]

        if torch.cuda.is_available():
            best_model = nn.DataParallel(best_model, device_ids=device_ids).cuda()
        else:
            best_model.to(device)

        if isinstance(best_model, nn.DataParallel):
            best_model.module.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)), strict=False)
        else:
            best_model.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)), strict=False) 
            best_model.to(device)
        
        test_dataloader = create_dataloaders(
            tokenizer, 
            test_data, 
            max_length, 
            inference_batch_size, 
            property_value=property, 
            pooling=pooling,
            normalize=False,
            preprocessing_strategy=preprocessing_strategy
        )
         
        predictions_list, test_performance = evaluate(best_model, mae_loss_function, test_dataloader, train_labels_mean, train_labels_std, property, device, task_name, normalizer=normalizer_type)
        predictions.append(predictions_list)
        test_results.append(test_performance)
    
    averaged_predictions = np.mean(np.array(predictions), axis=0)
    averaged_loss = np.mean(np.array(test_results))
    confidence_score = stdev(np.array(test_results))

    print('#'*50)
    print('The averaged test performance over 5 runs:', averaged_loss)
    print('The standard deviation:', confidence_score)
    print('#'*50)

    # save the averaged predictions
    test_predictions = {f"material_id":list(test_data['material_id']), f"actual_{property}":list(test_data[property]), f"predicted_{property}":averaged_predictions}
    saveCSV(pd.DataFrame(test_predictions), f"{results_path}/{model_name}_test_stats_for_{property}_{task_name}_{input_type}_{preprocessing_strategy}.csv")
    