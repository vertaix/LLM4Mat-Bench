import pandas as pd
import numpy as np
import json
import re

# for metrics
from torchmetrics.classification import BinaryAUROC
from sklearn import metrics

from create_args_parser import *

def readJSON(input_file):
    """
    1. arguments
        input_file: a json file to read
    2. output
        a json objet in a form of a dictionary
    """
    with open(input_file, "r", encoding="utf-8", errors='ignore') as infile:
        json_object = json.load(infile, strict=False)
    return json_object

def contains_elements_and_matches(input_string, elements):
    matching_elements = [element for element in elements if element in input_string]
    return bool(matching_elements), matching_elements

def extract_values(sentence):
    chem_elts = ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts']
    contains_elements, matching_elements = contains_elements_and_matches(sentence, chem_elts)
    
    # filter out chemical compound in answers
    if contains_elements:
        pattern = re.compile(rf'\b\w*{matching_elements[0]}\w*\b')
        sentence = re.sub(pattern, '', sentence)

    match_1 = re.search(r'(\d+(\.\d+)?)\s*x\s*10\s*\^*\s*(-?\d+)', sentence) # matching 2 x 10^6/2 x 10^-6values
    match_2 = re.search(r'(\d+(\.\d+)?)\s*×\s*10\s*\^*\s*(-?\d+)', sentence) # matching 2 × 10^6/2 × 10^-6 values
    match_3 = re.search(r'(\d+(\.\d+)?[eE][+-]?\d+)', sentence) # match 1e6 or 1E-08

    if match_1:
        value, _, exponent = match_1.groups()
        value = float(value)
        exponent = int(exponent)
        result = value * 10**exponent
        if result >= 100000.0 or result <= -100000.0:
            result = None

    elif match_2:
        value, _, exponent = match_2.groups()
        value = float(value)
        exponent = int(exponent)
        result = value * 10**exponent
        if result >= 100000.0 or result <= -100000.0:
            result = None
        
    elif match_3:
        notation = match_3.group()
        result = float(notation)
        if result >= 100000.0 or result <= -100000.0:
            result = None

    else:
        if "10^" in sentence:
            pattern = re.compile(r'(?<=\^)[+-]?\d+')
            match = pattern.search(sentence)
            if match:
                exponent = int(match.group())
                result = 10**exponent
                if result >= 100000.0 or result <= -100000.0:
                        result = None
            else:
                result = None
        else:
            matches = re.findall(r'(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)', sentence) #matches "1.0-3.4"->[('1.0', '3.4')]
            if len(matches) > 1:
                numbers = [float(matches[0][i]) for i in range(len(matches[0]))]
                result = np.array(numbers).mean()
            else:
                matches = re.findall(r'-?\d+\.?\d*', sentence)
                numbers = [float(number) if '.' in number else int(number) for number in matches]
                if len(numbers) > 0:
                    result = numbers[0]
                    if result >= 100000.0 or result <= -100000.0:
                        result = None
                else:
                    result = None
    return result

def extract_mp_gap_direct_predictions(sentence):
    positive_predictions = ['True','true','yes','likely','is a direct gap','Yes']
    negative_predictions = ['False','false', 'indirect']
    matching_positive = [prediction for prediction in positive_predictions if prediction in sentence]
    matching_negative = [prediction for prediction in negative_predictions if prediction in sentence]
    
    if bool(matching_positive):
        result = 1.0
    elif bool(matching_negative):
        result = 0.0
    else:
        result = None
    return result

def extract_mp_stability_predictions(sentence):
    positive_predictions = [':stable', ':Stable',' stable', ' Stable','True','true','yes','likely}','Yes']
    negative_predictions = ['unstable', 'Unstable','False','false']
    matching_positive = [prediction for prediction in positive_predictions if prediction in sentence]
    matching_negative = [prediction for prediction in negative_predictions if prediction in sentence]
    
    if bool(matching_positive):
        result = 1.0
    elif bool(matching_negative):
        result = 0.0
    else:
        result = None
    return result

def extract_snumat_direct_predictions(sentence):
    sentence = sentence.replace('\n\"CuCNH2S2\": \"Indirect\",\n\"Ge\": \"Direct\",\n\"TlFe2S3\": \"Indirect\",\n\"GaCl3\": \"Indirect\",\n\"(Nb2Tl5(SCl2)4)2Cl2\": \"Indirect\"', '')
    positive_predictions = [' direct', ' Direct','"direct\"','"Direct\"', ':Direct', ':direct']
    negative_predictions = [' indirect', ' Indirect', '"indirect\"', '"Indirect\"',':Indirect', ':indirect']
    matching_positive = [prediction for prediction in positive_predictions if prediction in sentence]
    matching_negative = [prediction for prediction in negative_predictions if prediction in sentence]
    
    if bool(matching_positive):
        result = 1.0
    elif bool(matching_negative):
        result = 0.0
    else:
        result = None
    return result

def extract_snumat_direct_hse_predictions(sentence):
    sentence = sentence.replace('\n\"CuCNH2S2\": \"Indirect HSE\",\n\"Ge\": \"Direct HSE\",\n\"TlFe2S3\": \"Direct HSE\",\n\"GaCl3\": \"Indirect HSE\",\n\"(Nb2Tl5(SCl2)4)2Cl2\": \"Indirect HSE\"', '')
    positive_predictions = [' direct', ' Direct','"direct\"','"Direct\"', '\"Direct HSE\"', '\"direct HSE\"', ' direct HSE', ' Direct HSE', ':Direct', ':direct', ':Direct HSE', ':direct HSE']
    negative_predictions = [' indirect', ' Indirect', '"indirect\"', '"Indirect\"', 'indirect HSE', 'Indirect HSE',':Indirect', ':indirect', ':Indirect HSE', ':indirect HSE']
    matching_positive = [prediction for prediction in positive_predictions if prediction in sentence]
    matching_negative = [prediction for prediction in negative_predictions if prediction in sentence]
    
    if bool(matching_positive):
        result = 1.0
    elif bool(matching_negative):
        result = 0.0
    else:
        result = None
    return result

def extract_snumat_soc_predictions(sentence):
    positive_predictions = ['True','true','yes','likely}','Yes']
    negative_predictions = ['False','false']
    matching_positive = [prediction for prediction in positive_predictions if prediction in sentence]
    matching_negative = [prediction for prediction in negative_predictions if prediction in sentence]
    
    if bool(matching_positive):
        result = 1.0
    elif bool(matching_negative):
        result = 0.0
    else:
        result = None
    return result 

def extract_predictions(dataset_name, model_name, data_path, results_path, input_type, prompt_type, property_name, max_len, batch_size, min_samples):
    print(f"Results on {dataset_name}:\n")

    data = pd.read_csv(f"{data_path}/{dataset_name}/{dataset_name}_inference_prompts_data.csv")

    data_dp = data.dropna(subset=[property_name])
    predictions = readJSON(f"{results_path}/{dataset_name}/{model_name}_test_stats_for_{property_name}_{input_type}_{prompt_type}_{max_len}_{batch_size}.json")

    results_df = pd.DataFrame({f'{property_name}_target': list(data_dp[property_name]), f'{property_name}_predicted': predictions})
    print(f'original results for {property_name}:', len(results_df))
    
    results_df = results_df[~results_df.isin([np.inf, -np.inf]).any(axis=1)]
    results_df = results_df.dropna(subset=[f'{property_name}_target']).reset_index(drop=True)
    print(f'after dropping target nans:', len(results_df)) 

    results_df[f'{property_name}_predicted'] = results_df[f'{property_name}_predicted'].replace('', pd.NA)
    results_df = results_df.dropna(subset=[f'{property_name}_predicted']).reset_index(drop=True) 
    print(f'after dropping predicted nans:', len(results_df))

    if dataset_name == 'mp': 
        if property_name == 'is_gap_direct':
            results_df[f'{property_name}_extracted_predictions'] = results_df[f'{property_name}_predicted'].apply(extract_mp_gap_direct_predictions)
            results_df = results_df.dropna(subset=[f'{property_name}_extracted_predictions']).reset_index(drop=True)
            print(f'after extracting predictions:', len(results_df)) 
            if len(results_df) >= min_samples:
                roc_score = metrics.roc_auc_score(list(results_df[f'{property_name}_target']), list(results_df[f'{property_name}_extracted_predictions']))
                print('ROC Score: ', roc_score)
            else:
                print('Invalid')
        elif property_name == 'is_stable':
            results_df[f'{property_name}_extracted_predictions'] = results_df[f'{property_name}_predicted'].apply(extract_mp_stability_predictions)
            results_df = results_df.dropna(subset=[f'{property_name}_extracted_predictions']).reset_index(drop=True)
            print(f'after extracting predictions:', len(results_df)) 
            if len(results_df) >= min_samples:
                roc_score = metrics.roc_auc_score(list(results_df[f'{property_name}_target']), list(results_df[f'{property_name}_extracted_predictions']))
                print('ROC Score: ', roc_score)
            else:
                print('Invalid')
        else:
            results_df[f'{property_name}_extracted_predictions'] = results_df[f'{property_name}_predicted'].apply(extract_values)
            results_df = results_df.dropna(subset=[f'{property_name}_extracted_predictions']).reset_index(drop=True)
            print(f'after extracting predictions:', len(results_df)) 
            print('max: ', results_df[f'{property_name}_extracted_predictions'].max())
            print('min: ', results_df[f'{property_name}_extracted_predictions'].min())

            if len(results_df) >= min_samples:
                mae = metrics.mean_absolute_error(list(results_df[f'{property_name}_target']), list(results_df[f'{property_name}_extracted_predictions']))
                rmse = metrics.mean_squared_error(list(results_df[f'{property_name}_target']), list(results_df[f'{property_name}_extracted_predictions']), squared=False)
                print('MAE: ', mae)
                print('RMSE: ', rmse)
            else:
                print('Invalid')

    elif dataset_name == 'snumat':
        results_df = results_df.drop(results_df[results_df[f'{property_name}_target'] == 'Null'].index).reset_index(drop=True)
        results_df.loc[results_df[f'{property_name}_target'] == "Direct", f'{property_name}_target'] = 1.0
        results_df.loc[results_df[f'{property_name}_target'] == "Indirect", f'{property_name}_target'] = 0.0
        results_df[f'{property_name}_target'] = results_df[f'{property_name}_target'].astype(float)

        if property_name == 'Direct_or_indirect':
            results_df[f'{property_name}_extracted_predictions'] = results_df[f'{property_name}_predicted'].apply(extract_snumat_direct_predictions)
            results_df = results_df.dropna(subset=[f'{property_name}_extracted_predictions']).reset_index(drop=True)
            print(f'after extracting predictions:', len(results_df)) 
            if len(results_df) >= min_samples:
                roc_score = metrics.roc_auc_score(list(results_df[f'{property_name}_target']), list(results_df[f'{property_name}_extracted_predictions']))
                print('ROC Score: ', roc_score)
            else:
                print("Invalid")
        elif property_name == 'Direct_or_indirect_HSE':
            results_df[f'{property_name}_extracted_predictions'] = results_df[f'{property_name}_predicted'].apply(extract_snumat_direct_hse_predictions)
            results_df = results_df.dropna(subset=[f'{property_name}_extracted_predictions']).reset_index(drop=True)
            print(f'after extracting predictions:', len(results_df)) 
            if len(results_df) >= min_samples:
                roc_score = metrics.roc_auc_score(list(results_df[f'{property_name}_target']), list(results_df[f'{property_name}_extracted_predictions']))
                print('ROC Score: ', roc_score)
            else:
                print("Invalid")
        elif property_name == "SOC":
            results_df[f'{property_name}_extracted_predictions'] = results_df[f'{property_name}_predicted'].apply(extract_snumat_soc_predictions)
            results_df = results_df.dropna(subset=[f'{property_name}_extracted_predictions']).reset_index(drop=True)
            print(f'after extracting predictions:', len(results_df)) 
            if len(results_df) >= min_samples: 
                roc_score = metrics.roc_auc_score(list(results_df[f'{property_name}_target']), list(results_df[f'{property_name}_extracted_predictions']))
                print('ROC Score: ', roc_score)
            else:
                print("Invalid")
        else:
            results_df[f'{property_name}_extracted_predictions'] = results_df[f'{property_name}_predicted'].apply(extract_values)
            results_df = results_df.dropna(subset=[f'{property_name}_extracted_predictions']).reset_index(drop=True)
            print(f'after extracting predictions:', len(results_df)) 
            print('max: ', results_df[f'{property_name}_extracted_predictions'].max())
            print('min: ', results_df[f'{property_name}_extracted_predictions'].min())

            if len(results_df) >= min_samples:
                mae = metrics.mean_absolute_error(list(results_df[f'{property_name}_target']), list(results_df[f'{property_name}_extracted_predictions']))
                rmse = metrics.mean_squared_error(list(results_df[f'{property_name}_target']), list(results_df[f'{property_name}_extracted_predictions']), squared=False)
                print('MAE: ', mae)
                print('RMSE: ', rmse)
            else:
                print("Invalid")
    
    else:
        results_df[f'{property_name}_extracted_predictions'] = results_df[f'{property_name}_predicted'].apply(extract_values)
        results_df = results_df.dropna(subset=[f'{property_name}_extracted_predictions']).reset_index(drop=True)
        print(f'after extracting predictions:', len(results_df)) 
        print('max: ', results_df[f'{property_name}_extracted_predictions'].max())
        print('min: ', results_df[f'{property_name}_extracted_predictions'].min())

        if len(results_df) >= min_samples:
            mae = metrics.mean_absolute_error(list(results_df[f'{property_name}_target']), list(results_df[f'{property_name}_extracted_predictions']))
            rmse = metrics.mean_squared_error(list(results_df[f'{property_name}_target']), list(results_df[f'{property_name}_extracted_predictions']), squared=False)
            print('MAE: ', mae)
            print('RMSE: ', rmse)
        else:
            print("Invalid")

    results_df.to_csv(f"{results_path}/{dataset_name}/{model_name}_test_stats_for_{property_name}_{input_type}_{prompt_type}_{max_len}_{batch_size}.csv", index=False)
    print('-'*50)
        
if __name__=='__main__':
    # set parameters
    args = args_parser()
    config = vars(args)
    
    dataset_name = config.get('dataset_name')
    input_type = config.get('input_type')
    prompt_type = config.get('prompt_type')
    batch_size = config.get("batch_size")
    max_len = config.get('max_len')
    property_name = config.get("property_name")
    model_name = config.get("model_name")
    data_path = config.get("data_path")
    results_path = config.get("results_path")
    min_samples = config.get("min_samples")

    extract_predictions(dataset_name, model_name, data_path, results_path, input_type, prompt_type, property_name, max_len, batch_size, min_samples)
    print(f"Done!")