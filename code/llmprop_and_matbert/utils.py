import re
import json
import glob
import torch
import tarfile
import datetime

# for metrics
from torchmetrics.classification import BinaryAUROC
from sklearn.metrics import roc_auc_score

# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# # Download stopwords if not already present
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

# def remove_nltk_stopwords(sentence):
#     stop_words = set(stopwords.words('english'))
#     # word_tokens = word_tokenize(sentence)
#     word_tokens = sentence.split()
#     filtered_sentence = [word for word in word_tokens if word.lower() not in stop_words]
#     return ' '.join(filtered_sentence).strip()

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

def writeTEXT(data, where_to_save):
    with open(where_to_save, "w", encoding="utf-8") as outfile:
        for d in data:
            outfile.write(str(d))
            outfile.write("\n")

def readTEXT_to_LIST(input_file):
    with open(input_file, "r", encoding="utf-8") as infile:
        data = []
        for line in infile:
            data.append(line)
    return data

def saveCSV(df, where_to_save):
    df.to_csv(where_to_save, index=False)

def time_format(total_time):
    """
    Change the from seconds to hh:mm:ss
    """
    total_time_rounded = int(round((total_time)))
    total_time_final = str(datetime.timedelta(seconds=total_time_rounded))
    return total_time_final

def z_normalizer(labels):
    """ Implement a z-score normalization technique"""
    labels_mean = torch.mean(labels)
    labels_std = torch.std(labels)

    scaled_labels = (labels - labels_mean) / labels_std

    return scaled_labels

def z_denormalize(scaled_labels, labels_mean, labels_std):
    labels = (scaled_labels * labels_std) + labels_mean
    return labels

def min_max_scaling(labels):
    """ Implement a min-max normalization technique"""
    min_val = torch.min(labels)
    max_val = torch.max(labels)
    diff = max_val - min_val
    scaled_labels = (labels - min_val) / diff
    return scaled_labels

def mm_denormalize(scaled_labels, min_val, max_val):
    diff = max_val - min_val
    denorm_labels = (scaled_labels * diff) + min_val
    return denorm_labels

def log_scaling(labels):
    """ Implement log-scaling normalization technique"""
    scaled_labels = torch.log1p(labels)
    return scaled_labels

def ls_denormalize(scaled_labels):
    denorm_labels = torch.expm1(scaled_labels)
    return denorm_labels

def compressCheckpointsWithTar(filename):
    filename_for_tar = filename[0:-3]
    tar = tarfile.open(f"{filename_for_tar}.tar.gz", "w:gz")
    tar.add(filename)
    tar.close()

def decompressTarCheckpoints(tar_filename):
    tar = tarfile.open(tar_filename)
    tar.extractall()
    tar.close()

def replace_bond_lengths_with_num(sentence):
    sentence = re.sub(r"\d+(\.\d+)?(?:–\d+(\.\d+)?)?\s*Å", "[NUM]", sentence) # Regex pattern to match bond lengths and units
    return sentence.strip()

def replace_bond_angles_with_ang(sentence):
    sentence = re.sub(r"\d+(\.\d+)?(?:–\d+(\.\d+)?)?\s*°", "[ANG]", sentence) # Regex pattern to match angles and units
    sentence = re.sub(r"\d+(\.\d+)?(?:–\d+(\.\d+)?)?\s*degrees", "[ANG]", sentence) # Regex pattern to match angles and units
    return sentence.strip()

def replace_bond_lengths_and_angles_with_num_and_ang(sentence):
    sentence = re.sub(r"\d+(\.\d+)?(?:–\d+(\.\d+)?)?\s*Å", "[NUM]", sentence) # Regex pattern to match bond lengths and units
    sentence = re.sub(r"\d+(\.\d+)?(?:–\d+(\.\d+)?)?\s*°", "[ANG]", sentence) # Regex pattern to match angles and units
    sentence = re.sub(r"\d+(\.\d+)?(?:–\d+(\.\d+)?)?\s*degrees", "[ANG]", sentence) # Regex pattern to match angles and units
    return sentence.strip()

def replace_numbers_with_num(sentence):
    sentence = re.sub(r'(\d+\.\d+|\d+)', '[NUM]', sentence)
    return sentence.strip()

def remove_bond_lengths_and_angles(sentence):
    sentence = re.sub(r"\d+(\.\d+)?(?:–\d+(\.\d+)?)?\s*Å", "", sentence) # Regex pattern to match bond lengths and units
    sentence = re.sub(r"\d+(\.\d+)?(?:–\d+(\.\d+)?)?\s*°", "", sentence) # Regex pattern to match angles and units
    sentence = re.sub(r"\d+(\.\d+)?(?:–\d+(\.\d+)?)?\s*degrees", "", sentence) # Regex pattern to match angles and units
    return sentence.strip()

def get_cleaned_stopwords():
    # from https://github.com/igorbrigadir/stopwords
    stopword_files = glob.glob("/n/fs/rnspace/projects/vertaix/LLM-Prop/stopwords/en/*.txt")
    num_str = {'one','two','three','four','five','six','seven','eight','nine'}

    all_stopwords_list = set()

    for file_path in stopword_files:
        all_stopwords_list |= set(readTEXT_to_LIST(file_path))

    cleaned_list_for_mat = {wrd.replace("\n", "").strip() for wrd in all_stopwords_list} - {wrd for wrd in all_stopwords_list if wrd.isdigit()} - num_str
    
    return cleaned_list_for_mat

def remove_mat_stopwords(sentence):
    stopwords_list = get_cleaned_stopwords()
    words = sentence.split()
    words_lower = sentence.lower().split()
    sentence = ' '.join([words[i] for i in range(len(words)) if words_lower[i] not in stopwords_list])
    return sentence

def get_sequence_len_stats(df, tokenizer, max_len, input_type):
    training_on = sum(1 for sent in df[input_type].apply(tokenizer.tokenize) if len(sent) <= max_len)
    return (training_on/len(df))*100

def get_max_len(df, tokenizer, input_type):
    max_len = max(len(sent) for sent in df[input_type].apply(tokenizer.tokenize))
    return max_len

def get_numbers_in_a_sentence(sentence):
    pattern = r'(\d+\.\d+|\d+)'
    matches = re.findall(pattern, sentence)
    numbers = [float(number) if '.' in number else int(number) for number in matches]
    return numbers

def normalize_values(list_of_lists, min_value=-5, max_value=5):
    flattened_list = [item for sublist in list_of_lists for item in sublist]
    min_val = min(flattened_list)
    max_val = max(flattened_list)
    normalized_lists = [
        [
            min_value + (max_value - min_value) * (value - min_val) / (max_val - min_val)
            for value in sublist
        ]
        for sublist in list_of_lists
    ]
    return normalized_lists

def get_roc_score(predictions, targets):
    roc_fn = BinaryAUROC(threshold=None)
    x = torch.tensor(targets)
    y = torch.tensor(predictions)
    y = torch.round(torch.sigmoid(y))
    roc_score = roc_fn(y, x)
    return roc_score