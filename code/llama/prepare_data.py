import os
import time
import json
import argparse
import pandas as pd
from transformers import AutoTokenizer

from huggingface_hub import login
token = "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"
login(token)

def writeToJSON(data, where_to_save):
    """
    data: a dictionary that contains data to save
    where_to_save: the name of the file to write on
    """
    with open(where_to_save, "w", encoding="utf8") as outfile:
        json.dump(data, outfile)
        
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

system_prompt_comp = """You are a material scientist. 
Look at the chemical formula of the given crystalline material and predict its property.
The output must be in a json format. For example: {property_name:predicted_property_value}.
Answer as precise as possible and in few words as possible."""

system_prompt_struct = """
You are a material scientist. 
Look at the cif structure information of the given crystalline material and predict its property.
The output must be in a json format. For example: {property_name:predicted_property_value}.
Answer as precise as possible and in few words as possible.
"""

system_prompt_descr = """
You are a material scientist. 
Look at the structure description of the given crystalline material and predict its property.
The output must be in a json format. For example: {property_name:predicted_property_value}.
Answer as precise as possible and in few words as possible.
"""

def zs_comp(sentence, property_name=''):
    input_prompt = f"""chemical formula: {sentence}.\nproperty name: {property_name}."""
    prompt_template = f"""<s>[INST] <<SYS>>\n{system_prompt_comp}\n<</SYS>>\n\n{input_prompt} [/INST]"""
    return prompt_template

def zs_descr(sentence, property_name='', max_words=50):
    sentence = ' '.join(str(sentence).split()[0:max_words]) if len(str(sentence).split())>max_words else sentence
    input_prompt = f"""structure description: {sentence}.\nproperty name: {property_name}."""
    prompt_template = f"""<s>[INST] <<SYS>>\n{system_prompt_descr}\n<</SYS>>\n\n{input_prompt} [/INST]"""
    return prompt_template

def zs_struct(sentence, property_name='', max_words=50):
    sentence = ' '.join(str(sentence).split()[0:max_words]) if len(str(sentence).split())>max_words else sentence
    input_prompt = f"""cif structure: {sentence}.\nproperty name: {property_name}."""
    prompt_template = f"""<s>[INST] <<SYS>>\n{system_prompt_struct}\n<</SYS>>\n\n{input_prompt} [/INST]"""
    return prompt_template

def fs_comp(sentence, train, property_name=''):
    train_top_5_df  = train.head(5)
    five_shot_prompt = "\n".join([f'chemical formula: {train["formula"][i]}.\nproperty name: {property_name}.\n' + '{'+ f'{property_name}:{train[property_name][i]}' +'}.\n' for i in range(5)])
    input_prompt = f"""{five_shot_prompt}\nchemical formula: {sentence}.\nproperty name: {property_name}."""
    prompt_template = f"""<s>[INST] <<SYS>>\n{system_prompt_comp}\n<</SYS>>\n\n{input_prompt} [/INST]"""
    return prompt_template

def fs_descr(sentence, train, property_name='', max_words=10):
    train_top_5_df  = train.head(5)
    train['truncated_description'] = train_top_5_df['description'].apply(lambda x: ' '.join(str(x).split()[0:10]) if len(str(x).split())>10 else x)
    sentence = ' '.join(str(sentence).split()[0:max_words]) if len(str(sentence).split())>max_words else sentence
    five_shot_prompt = "\n".join([f'structure description: {train["truncated_description"][i]}.\nproperty name: {property_name}.\n' + '{'+ f'{property_name}:{train[property_name][i]}' +'}.\n' for i in range(5)])
    input_prompt = f"""{five_shot_prompt}\nstructure description: {sentence}.\nproperty name: {property_name}."""
    prompt_template = f"""<s>[INST] <<SYS>>\n{system_prompt_descr}\n<</SYS>>\n\n{input_prompt} [/INST]"""
    return prompt_template

def fs_struct(sentence, train, property_name='', max_words=10):
    train_top_5_df  = train.head(5)
    train['truncated_structure'] = train_top_5_df['cif_structure'].apply(lambda x: ' '.join(str(x).split()[0:10]) if len(str(x).split())>10 else x)
    sentence = ' '.join(str(sentence).split()[0:max_words]) if len(str(sentence).split())>max_words else sentence
    five_shot_prompt = "\n".join([f'cif structure: {train["truncated_structure"][i]}.\nproperty name: {property_name}.\n' + '{'+ f'{property_name}:{train[property_name][i]}' +'}.\n' for i in range(5)])
    input_prompt = f"""{five_shot_prompt}\ncif structure: {sentence}.\nproperty name: {property_name}."""
    prompt_template = f"""<s>[INST] <<SYS>>\n{system_prompt_struct}\n<</SYS>>\n\n{input_prompt} [/INST]"""
    return prompt_template

def generate_prompt(data_path, dataset_name, tokenizer, zs_max_words, fs_max_words):
    """
    0. Get its df_train and df_test 
    1. Generate zs and 5s for composition, description, and structure for each property. 
        e.g: bg_comp_zs, bg_comp_fs, bg_descr_zs, bg_descr_fs, bg_struct_zs, bg_struct_fs, bg_value
    2. For descr and struct fs examples, pick the top 5 shorterst examples from df_train and use them as fs examples
    3. Ensure that each prompt is within 4000 tokens:
        e.g: use llama tokenizer to truncate the description or structure string to fit within the length limit
    4. Save each dataset as a whole separately
    """
   
    train = pd.read_csv(f"{data_path}/{dataset_name}/unfiltered/train.csv")
    test = pd.read_csv(f"{data_path}/{dataset_name}/unfiltered/test.csv")

    if dataset_name == "mp":
        test = test.rename(columns={"formula_pretty":"formula"})
        train = train.rename(columns={"formula_pretty":"formula"})
    
    input_prompt = ''
    data = {}

    if dataset_name == "mp":
        property_names = ["band_gap","volume","is_gap_direct","formation_energy_per_atom","energy_above_hull","energy_per_atom","is_stable","density","density_atomic","efermi"]
    elif dataset_name == "hea":
        property_names = ["Ef_per_atom","e_above_hull","volume_per_atom","e_per_atom"]
    elif dataset_name == "snumat":
        property_names = ["Band_gap_HSE","Band_gap_GGA","Band_gap_GGA_optical","Band_gap_HSE_optical", "Direct_or_indirect","Direct_or_indirect_HSE","SOC","Direct_or_indirect"]
    elif dataset_name == "gnome":
        property_names = ["Formation_Energy_Per_Atom", "Decomposition_Energy_Per_Atom", "Bandgap", "Corrected_Energy", "Volume", "Density"]
    elif dataset_name == "hmof":
        property_names = ["max_co2_adsp", "min_co2_adsp", "lcd", "pld", "void_fraction", "surface_area_m2g", "surface_area_m2cm3"]
    elif dataset_name == "omdb":
        property_names = ["bandgap"]
    elif dataset_name == "oqmd":
        property_names = ["bandgap","e_form"]
    elif dataset_name == "qe_tb":
        property_names = ["energy_per_atom", "indir_gap", "f_enp", "final_energy"]
    elif dataset_name == "qmof":
        property_names = ["energy_total", "bandgap", "lcd", "pld"]
    elif dataset_name == "jarvis":
        property_names = ["formation_energy_peratom","optb88vdw_bandgap","slme","spillage","optb88vdw_total_energy","mepsx","max_efg","avg_elec_mass","dfpt_piezo_max_eij","dfpt_piezo_max_dij","dfpt_piezo_max_dielectric","n-Seebeck","n-powerfact","p-Seebeck","p-powerfact","exfoliation_energy","bulk_modulus_kv","shear_modulus_gv","mbj_bandgap","ehull"]
    
    train = train.dropna(subset=property_names).reset_index(drop=True)

    str_to_remove_from_struct = "# generated using pymatgen\n"
    train['cif_structure'] = train['cif_structure'].apply(lambda x: str(x).replace(str_to_remove_from_struct, ''))
    test['cif_structure'] = test['cif_structure'].apply(lambda x: str(x).replace(str_to_remove_from_struct, ''))

    for property_name in property_names:
        for input_type in ['formula','description','cif_structure']: 
            for prompt_type in ['zero_shot', 'few_shot']:
                if prompt_type == "zero_shot": 
                    if input_type == 'formula':
                        data[f"{property_name}_{input_type}_{prompt_type}"] = test[input_type].apply(lambda x: zs_comp(x, property_name=property_name)) 
                    elif input_type == 'description':
                        data[f"{property_name}_{input_type}_{prompt_type}"] = test[input_type].apply(lambda x: zs_descr(x, property_name=property_name, max_words=zs_max_words))
                    elif input_type == 'cif_structure':
                        data[f"{property_name}_{input_type}_{prompt_type}"] = test[input_type].apply(lambda x: zs_struct(x, property_name=property_name, max_words=zs_max_words))
                elif prompt_type == 'few_shot':
                    if input_type == 'formula':
                        data[f"{property_name}_{input_type}_{prompt_type}"] = test[input_type].apply(lambda x: fs_comp(x, train, property_name=property_name))
                    elif input_type == 'description':
                        data[f"{property_name}_{input_type}_{prompt_type}"] = test[input_type].apply(lambda x: fs_descr(x, train, property_name=property_name, max_words=zs_max_words))
                    elif input_type == 'cif_structure':
                        data[f"{property_name}_{input_type}_{prompt_type}"] = test[input_type].apply(lambda x: fs_struct(x, train, property_name=property_name, max_words=zs_max_words))

        data[f"{property_name}"] = list(test[property_name])
        print(f"Finished {property_name} data")
        print('-'*50)
    data_df = pd.DataFrame(data)
    data_df.to_csv(f"{data_path}/{dataset_name}/unfiltered/{dataset_name}_inference_prompts_data.csv", index=False)

if __name__=='__main__':
    # parse Arguments
    parser = argparse.ArgumentParser(description='prepare_prompts')
    parser.add_argument('--dataset_name',
                        help='Any dataset name that is in LLM4MatBench',
                        type=str,
                        default='mp')
    parser.add_argument('--zs_max_words',
                        help='input sequence length limit before tokenization',
                        type=int,
                        default=50)
    parser.add_argument('--fs_max_words',
                        help='input sequence length limit before tokenization',
                        type=int,
                        default=10)
    parser.add_argument('--data_path',
                        help='A path tom load data from and also save prompts',
                        type=str,
                        default="")
    args = parser.parse_args()
    config = vars(args)
    
    data_path = config.get('data_path')
    dataset_name = config.get('dataset_name')
    zs_max_words = config.get('zs_max_words')
    fs_max_words = config.get('fs_max_words')
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    
    # ["mp","oqmd","qe_tb","jarvis","qmof","hea","snumat","gnome","hmof","omdb"]
    # dataset_path = f"{data_path}/{dataset_name}/"
    # if not os.path.exists(dataset_path):
    #     os.makedirs(dataset_path)
    generate_prompt(data_path, dataset_name,  tokenizer, zs_max_words, fs_max_words)
    print("Finished to process the data for ", dataset_name)
    print('@'*50)
