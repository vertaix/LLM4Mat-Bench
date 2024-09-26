import os
import time
import pandas as pd
import json
from transformers import AutoTokenizer

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

def generate_prompt_new(tokenizer, dataset_name):
    """
    0. Get its df_train and df_test 
    1. Generate zs and 5s for composition, description, and structure for each property. 
        e.g: bg_comp_zs, bg_comp_fs, bg_descr_zs, bg_descr_fs, bg_struct_zs, bg_struct_fs, bg_value
    2. For descr and struct fs examples, pich the top 5 shorterst examples from df_train and use them as fs examples
    3. Ensure that each prompt is within 4000 tokens:
        e.g: use llama tokenizer to truncate the description or structure string to fit within the length limit
    4. Save each dataset as a whole separately
    """
    input_path = "/n/fs/rnspace/projects/vertaix/nlp4matbench/data"
    train = pd.read_csv(f"{input_path}/{dataset_name}/unfiltered/train.csv")

    test = pd.read_csv(f"{input_path}/{dataset_name}/unfiltered/test.csv")

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
        for input_type in ['description','cif_structure']: 
            for prompt_type in ['zero_shot', 'few_shot']:
                if prompt_type == "zero_shot": 
                    if input_type == 'description':
                        data[f"{property_name}_{input_type}_{prompt_type}"] = test[input_type].apply(lambda x: zs_descr(x, property_name=property_name))
                    elif input_type == 'cif_structure':
                        data[f"{property_name}_{input_type}_{prompt_type}"] = test[input_type].apply(lambda x: zs_struct(x, property_name=property_name))
                elif prompt_type == 'few_shot':
                    if input_type == 'description':
                        data[f"{property_name}_{input_type}_{prompt_type}"] = test[input_type].apply(lambda x: fs_descr(x, train, property_name=property_name))
                    elif input_type == 'cif_structure':
                        data[f"{property_name}_{input_type}_{prompt_type}"] = test[input_type].apply(lambda x: fs_struct(x, train, property_name=property_name))

    
        data[f"{property_name}"] = list(test[property_name])
        print(f"Finished {property_name} data")
        print('-'*50)
    data_df = pd.DataFrame(data)
    data_df.to_csv(f"{input_path}/{dataset_name}/unfiltered/{dataset_name}_prompting_data_chat_struct_and_descr.csv", index=False)

def generate_prompt(tokenizer, dataset_name):
    """
    0. Get its df_train and df_test 
    1. Generate zs and 5s for composition, description, and structure for each property. 
        e.g: bg_comp_zs, bg_comp_fs, bg_descr_zs, bg_descr_fs, bg_struct_zs, bg_struct_fs, bg_value
    2. For descr and struct fs examples, pich the top 5 shorterst examples from df_train and use them as fs examples
    3. Ensure that each prompt is within 4000 tokens:
        e.g: use llama tokenizer to truncate the description or structure string to fit within the length limit
    4. Save each dataset as a whole separately
    """
    input_path = "/n/fs/rnspace/projects/vertaix/nlp4matbench/data"
    train = pd.read_csv(f"{input_path}/{dataset_name}/unfiltered/train.csv")

    test = pd.read_csv(f"{input_path}/{dataset_name}/unfiltered/test.csv")

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

    for prompt_type in ['zero_shot', 'few_shot']:
        for input_type in ["cif_structure", "description"]: #
            for property_name in property_names:
                prompts = []
                
                if prompt_type == "zero_shot" and input_type == "formula":
                    for i in range(len(test)):
                        input_prompt = f"""chemical formula: {test['formula'][i]}.\nproperty name: {property_name}."""
                        prompt_template = f"""<s>[INST] <<SYS>>\n{system_prompt_comp}\n<</SYS>>\n\n{input_prompt} [/INST]"""
                        prompts.append(prompt_template)

                elif prompt_type == "zero_shot" and input_type == "description":
                    for i in range(len(test)):
                        test['truncated_description'] = test['description'].apply(lambda x: ' '.join(str(x).split()[0:50]) if len(str(x).split())>50 else x)
                        input_prompt = f"""structure description: {test['truncated_description'][i]}.\nproperty name: {property_name}."""
                        prompt_template = f"""<s>[INST] <<SYS>>\n{system_prompt_descr}\n<</SYS>>\n\n{input_prompt} [/INST]"""
                        prompts.append(prompt_template)

                elif prompt_type == "zero_shot" and input_type == "cif_structure":
                    for i in range(len(test)):
                        test['truncated_structure'] = test['cif_structure'].apply(lambda x: ' '.join(x.split()[0:50]) if len(x.split())>50 else x)
                        input_prompt = f"""cif structure: {test['truncated_structure'][i]}.\nproperty name: {property_name}."""
                        prompt_template = f"""<s>[INST] <<SYS>>\n{system_prompt_struct}\n<</SYS>>\n\n{input_prompt} [/INST]"""
                        prompts.append(prompt_template)

                elif prompt_type == "few_shot":
                    if input_type == "formula":
                        for i in range(len(test)):
                            five_shot_prompt = "\n".join([f'chemical formula: {train["formula"][i]}.\nproperty name: {property_name}.\n' + '{'+ f'{property_name}:{train[property_name][i]}' +'}.\n' for i in range(5)])
                            input_prompt = f"""{five_shot_prompt}\nchemical formula: {test['formula'][i]}.\nproperty name: {property_name}."""
                            prompt_template = f"""<s>[INST] <<SYS>>\n{system_prompt_comp}\n<</SYS>>\n\n{input_prompt} [/INST]"""
                            prompts.append(prompt_template)

                    elif input_type == "description":
                        for i in range(len(test)):
                            train['truncated_description'] = train['description'].apply(lambda x: ' '.join(str(x).split()[0:10]) if len(str(x).split())>10 else x)
                            test['truncated_description'] = test['description'].apply(lambda x: ' '.join(str(x).split()[0:10]) if len(str(x).split())>10 else x)
                            five_shot_prompt = "\n".join([f'structure description: {train["truncated_description"][i]}.\nproperty name: {property_name}.\n' + '{'+ f'{property_name}:{train[property_name][i]}' +'}.\n' for i in range(5)])
                            input_prompt = f"""{five_shot_prompt}\nstructure description: {test['truncated_description'][i]}.\nproperty name: {property_name}."""
                            prompt_template = f"""<s>[INST] <<SYS>>\n{system_prompt_descr}\n<</SYS>>\n\n{input_prompt} [/INST]"""
                            prompts.append(prompt_template)

                    elif input_type == "cif_structure":
                        for i in range(len(test)):
                            train['truncated_structure'] = train['cif_structure'].apply(lambda x: ' '.join(x.split()[0:10]) if len(x.split())>10 else x)
                            test['truncated_structure'] = test['cif_structure'].apply(lambda x: ' '.join(x.split()[0:10]) if len(x.split())>10 else x)
                            five_shot_prompt = "\n".join([f'cif structure: {train["truncated_structure"][i]}.\nproperty name: {property_name}.\n' + '{'+ f'{property_name}:{train[property_name][i]}' +'}.\n' for i in range(5)])
                            input_prompt = f"""{five_shot_prompt}\ncif structure: {test['truncated_structure'][i]}.\nproperty name: {property_name}."""
                            prompt_template = f"""<s>[INST] <<SYS>>\n{system_prompt_struct}\n<</SYS>>\n\n{input_prompt} [/INST]"""
                            prompts.append(prompt_template)

                data.update({f"{property_name}_{input_type}_{prompt_type}":prompts, f"{property_name}":list(test[property_name])})
                print(f"Finished {property_name} data")
                print('-'*50)
    data_df = pd.DataFrame(data)
    data_df.to_csv(f"{input_path}/{dataset_name}/unfiltered/{dataset_name}_prompting_data_chat_struct_and_descr.csv", index=False)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

for dataset_name in ["mp","oqmd","qe_tb","jarvis","qmof","hea","snumat","gnome","hmof","omdb"]:
    generate_prompt_new(tokenizer, dataset_name)
    print("Finished to process the data for ", dataset_name)
    print('@'*50)
