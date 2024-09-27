import pandas as pd
import numpy as np
import json
import re

# for metrics
from torchmetrics.classification import BinaryAUROC
from sklearn import metrics

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

def extract_predictions(dataset_name, data_path, results_path, input_type="formula", prompt_type="zero_shot"):
    print(f"Results on {dataset_name}:\n")
    
    max_len = 800
    batch_size = 8
    data = pd.read_csv(f"{data_path}/{dataset_name}/unfiltered/{dataset_name}_prompting_data_chat.csv")

    if dataset_name == "mp":
        property_names = ["formation_energy_per_atom","band_gap","energy_per_atom","energy_above_hull","efermi","volume","density","density_atomic","is_gap_direct","is_stable"] 
    elif dataset_name == "hea":
        property_names = ["Ef_per_atom","e_per_atom", "e_above_hull","volume_per_atom"]
    elif dataset_name == "snumat":
        property_names = ["Band_gap_GGA","Band_gap_HSE","Band_gap_GGA_optical","Band_gap_HSE_optical","Direct_or_indirect","Direct_or_indirect_HSE","SOC"]
    elif dataset_name == "gnome":
        property_names = ["Formation_Energy_Per_Atom","Bandgap", "Decomposition_Energy_Per_Atom", "Corrected_Energy", "Volume", "Density"] 
    elif dataset_name == "hmof":
        property_names = ["max_co2_adsp", "min_co2_adsp", "lcd", "pld", "void_fraction", "surface_area_m2g", "surface_area_m2cm3"]
    elif dataset_name == "omdb":
        property_names = ["bandgap"]
    elif dataset_name == "oqmd":
        property_names = ["e_form","bandgap"]
    elif dataset_name == "qe_tb":
        property_names = ["f_enp","energy_per_atom", "final_energy", "indir_gap"]
    elif dataset_name == "qmof":
        property_names = ["bandgap", "energy_total",  "lcd", "pld"]
    elif dataset_name == "jarvis":
        property_names = ["formation_energy_peratom","optb88vdw_bandgap","optb88vdw_total_energy","ehull","mbj_bandgap","bulk_modulus_kv","shear_modulus_gv","slme", "spillage","mepsx","dfpt_piezo_max_dielectric","dfpt_piezo_max_dij","dfpt_piezo_max_eij","max_efg","exfoliation_energy","avg_elec_mass","n-Seebeck","n-powerfact","p-Seebeck","p-powerfact"] 
    
    for property_name in property_names:
        data_dp = data.dropna(subset=[property_name])
        predictions = readJSON(f"{results_path}/{dataset_name}/llama_test_stats_for_{property_name}_{input_type}_{prompt_type}_{max_len}_{batch_size}.json")

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
                if len(results_df) >= 10:
                    roc_score = metrics.roc_auc_score(list(results_df[f'{property_name}_target']), list(results_df[f'{property_name}_extracted_predictions']))
                    print('ROC Score: ', roc_score)
                else:
                    print('Invalid')
            elif property_name == 'is_stable':
                results_df[f'{property_name}_extracted_predictions'] = results_df[f'{property_name}_predicted'].apply(extract_mp_stability_predictions)
                results_df = results_df.dropna(subset=[f'{property_name}_extracted_predictions']).reset_index(drop=True)
                print(f'after extracting predictions:', len(results_df)) 
                if len(results_df) >= 10:
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

                if len(results_df) >= 10:
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
                if len(results_df) >= 10:
                    roc_score = metrics.roc_auc_score(list(results_df[f'{property_name}_target']), list(results_df[f'{property_name}_extracted_predictions']))
                    print('ROC Score: ', roc_score)
                else:
                    print("Invalid")
            elif property_name == 'Direct_or_indirect_HSE':
                results_df[f'{property_name}_extracted_predictions'] = results_df[f'{property_name}_predicted'].apply(extract_snumat_direct_hse_predictions)
                results_df = results_df.dropna(subset=[f'{property_name}_extracted_predictions']).reset_index(drop=True)
                print(f'after extracting predictions:', len(results_df)) 
                if len(results_df) >= 10:
                    roc_score = metrics.roc_auc_score(list(results_df[f'{property_name}_target']), list(results_df[f'{property_name}_extracted_predictions']))
                    print('ROC Score: ', roc_score)
                else:
                    print("Invalid")
            elif property_name == "SOC":
                results_df[f'{property_name}_extracted_predictions'] = results_df[f'{property_name}_predicted'].apply(extract_snumat_soc_predictions)
                results_df = results_df.dropna(subset=[f'{property_name}_extracted_predictions']).reset_index(drop=True)
                print(f'after extracting predictions:', len(results_df)) 
                if len(results_df) >= 10: 
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

                if len(results_df) >= 10:
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

            if len(results_df) >= 10:
                mae = metrics.mean_absolute_error(list(results_df[f'{property_name}_target']), list(results_df[f'{property_name}_extracted_predictions']))
                rmse = metrics.mean_squared_error(list(results_df[f'{property_name}_target']), list(results_df[f'{property_name}_extracted_predictions']), squared=False)
                print('MAE: ', mae)
                print('RMSE: ', rmse)
            else:
                print("Invalid")

        results_df.to_csv(f"{results_path}/{dataset_name}/llama_test_stats_for_{property_name}_{input_type}_{prompt_type}_{max_len}_{batch_size}.csv", index=False)
        print('-'*50)
        
if __name__=='__main__':
    # parse Arguments
    parser = argparse.ArgumentParser(description='evaluate')
    parser.add_argument('--dataset_names',
                        help='A list of dataset names',
                        type=list,
                        default=['mp'])
    parser.add_argument('--data_path',
                        help='A path to prompts',
                        type=str,
                        default="")
    parser.add_argument('--results_path',
                        help='A path of where the results will be saved',
                        type=str,
                        default="")
    args = parser.parse_args()
    config = vars(args)
    
    dataset_names = config.get('dataset_names')
    data_path = config.get('data_path')
    results_path = config.get('results_path')

    for dataset_name in dataset_names:
        for input_type in ["description", "cif_structure", "formula"]:
            for prompt_type in ["zero_shot", "few_shot"]: 
                extract_predictions(dataset_name, data_path, results_path, input_type=input_type, prompt_type=prompt_type)
                print(f"Finished {prompt_type} results.")
                print("@"*50)
            print(f'Finished {input_type} results.')
            print("+"*50)