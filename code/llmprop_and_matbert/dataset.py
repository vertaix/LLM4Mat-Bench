"""
A function to prepare the dataloaders
"""
# Import packages
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils import *

np.random.seed(42)

def find_indices_of_pattern(input_list, pattern):
    return [i for i, item in enumerate(input_list) if item == pattern]

def get_x_num(max_length, txt_tokens_ids, num_token_id, numbers_in_txt_normalized):
    x_num = torch.ones((len(txt_tokens_ids), max_length)) 
    num_token_idx = [find_indices_of_pattern(txt_tokens_id, num_token_id) for txt_tokens_id in txt_tokens_ids]
    
    for n in range(len(num_token_idx)):
        if len(num_token_idx[n]) > 0:
            for i, id in enumerate(num_token_idx[n]):
                x_num[n][id] = numbers_in_txt_normalized[n][i]
        else:
            continue

    return x_num

def tokenize(tokenizer, dataframe, max_length, pooling='cls', input_type='description'):
    """
    1. Takes in the the list of input sequences and return 
    the input_ids and attention masks of the tokenized sequences
    2. max_length = the max length of each input sequence 
    """
    # dataframe['list_of_numbers_in_input'] = dataframe[input_type].apply(get_numbers_in_a_sentence)
    # dataframe[input_type] = dataframe[input_type].apply(replace_numbers_with_num)

    if pooling == 'cls':
        encoded_corpus = tokenizer(text=["[CLS] " + str(descr) for descr in dataframe[input_type].tolist()],
                                    add_special_tokens=True,
                                    padding='max_length',
                                    truncation='longest_first',
                                    max_length=max_length, # According to ByT5 paper
                                    return_attention_mask=True)
    elif pooling in ['mean', None]:
        encoded_corpus = tokenizer(text=dataframe[input_type].tolist(),
                                    add_special_tokens=True,
                                    padding='max_length',
                                    truncation='longest_first',
                                    max_length=max_length, # According to ByT5 paper
                                    return_attention_mask=True) 
    input_ids = encoded_corpus['input_ids']
    attention_masks = encoded_corpus['attention_mask']

    return input_ids, attention_masks

def create_dataloaders(
    tokenizer, 
    dataframe, 
    max_length, 
    batch_size, 
    property_value="band_gap", 
    pooling='cls', 
    normalize=False, 
    normalizer='z_norm', 
    input_type='description', 
    preprocessing_strategy=None
):
    """
    Dataloader which arrange the input sequences, attention masks, and labels in batches
    and transform the to tensors
    """
    input_ids, attention_masks = tokenize(tokenizer, dataframe, max_length, pooling=pooling, input_type=input_type)
    labels = dataframe[property_value].to_numpy()

    input_tensor = torch.tensor(input_ids)
    mask_tensor = torch.tensor(attention_masks)
    labels_tensor = torch.tensor(labels)
    
    if batch_size <= 0:
        raise ValueError(f"Invalid batch_size={batch_size}. Must be a positive integer.")

    if normalize:
        if normalizer == 'z_norm':
            normalized_labels = z_normalizer(labels_tensor)
        elif normalizer == 'mm_norm':
            normalized_labels = min_max_scaling(labels_tensor)
        elif normalizer == 'ls_norm':
            normalized_labels = log_scaling(labels_tensor)
        elif normalizer == 'no_norm':
            normalized_labels = labels_tensor

        if preprocessing_strategy == 'xVal':
            x_num = dataframe.list_of_numbers_in_input.tolist()
            num_token_id = tokenizer.convert_tokens_to_ids('[NUM]')
            x_num_tensor = get_x_num(max_length, input_ids, num_token_id, x_num)
            dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor, normalized_labels, x_num_tensor)
        else:
            dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor, normalized_labels)
    else:
        if preprocessing_strategy == 'xVal':
            x_num = dataframe.list_of_numbers_in_input.tolist()
            num_token_id = tokenizer.convert_tokens_to_ids('[NUM]')
            x_num_tensor = get_x_num(max_length, input_ids, num_token_id, x_num)
            dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor, x_num_tensor)
        else:
            dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # Set the shuffle to False for now since the labels are continues values check later if this may affect the result

    return dataloader
