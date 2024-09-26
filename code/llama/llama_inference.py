from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, AutoTokenizer, LlamaForSequenceClassification
import transformers
import torch
# import replicate
import os
import time
import pandas as pd
import json
import argparse

from huggingface_hub import login

login("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")

# token = "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"

def extract_ans_from_chat_llm(result):
    # Find the content within curly braces
    start_index = result.find('{')
    end_index = result.find('}')

    # Extract the content and format as JSON
    json_content = result[start_index:end_index + 1]
    return json_content

def extract_ans_from_next_token_llm(result):
    result = str(result)
    if len(result) != 0:
        output = result.split()
    else:
        output = result
    return output

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

def generate(model, tokenizer, prompts, max_len, batch_size):
    results = []
    pipe = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        batch_size=batch_size,
    )

    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

    sequences = pipe(
            prompts,
            do_sample=True,
            # temperature=0.1,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=max_len,
            return_full_text=False,
            # pad_token_id=tokenizer.eos_token_id,
        )

    for seqs in sequences:
        # print(seqs)
        for seq in seqs:
            # print(seq['generated_text'], "\n", "-"*50)
            # print(extract_ans(seq['generated_text']), "\n", "-"*50)
            results.append(extract_ans_from_chat_llm(seq['generated_text']))
    # for seqs in sequences:
    #     print(seqs, "\n", "-"*50)
    #     # print(extract_ans(seq['generated_text']), "\n", "-"*50)
    #     results.append(extract_ans_from_chat_llm(seqs['generated_text']))
    #     # results.append(seqs)
    
    return results

if __name__ == "__main__":
    # parse Arguments
    parser = argparse.ArgumentParser(description='llama_inference')
    parser.add_argument('--dataset_name',
                        help='Name of the dataset',
                        type=str,
                        default='mp')
    parser.add_argument('--input_type',
                        help='Type of input',
                        type=str,
                        default="formula")
    parser.add_argument('--batch_size',
                        help='Batch size',
                        type=int,
                        default=32)
    parser.add_argument('--prompt_type',
                        help='Type of the prompt',
                        type=str,
                        default="zero_shot")
    parser.add_argument('--property_name',
                        help='Name of the property',
                        type=str,
                        default="band_gap")
    parser.add_argument('--max_len',
                        help='Max output sequence length',
                        type=int,
                        default=200)
    parser.add_argument('--model_name',
                        help='Name of the model',
                        type=str,
                        default="llama")
    args = parser.parse_args()
    config = vars(args)

    # print(os.environ['TRANSFORMERS_CACHE'])
    print(os.environ['HF_HOME'])
    print(os.environ['HF_DATASETS_CACHE'])
    
    cache_directory = "/n/fs/rnspace/.hf_cache/"

    # check if the GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'Number of available devices: {torch.cuda.device_count()}')
        print(f'Current device is: {torch.cuda.current_device()}')
        print("Training and testing on", torch.cuda.device_count(), "GPUs!")
        print('-'*50)
    else:
        print("No GPU available, please connect to the GPU first or continue to use CPU instead")
        print('-'*50)
        device = torch.device("cpu")

    # set parameters

    dataset_name = config.get('dataset_name')
    input_type = config.get('input_type')
    prompt_type = config.get('prompt_type')
    batch_size = config.get("batch_size")
    max_len = config.get('max_len')
    property_name = config.get("property_name")
    model_name = config.get("model_name")

    statistics_directory = f"/n/fs/rnspace/projects/vertaix/nlp4matbench/statistics/{dataset_name}/"
    model = "meta-llama/Llama-2-7b-chat-hf" 
    # model = "daryl149/llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model, 
                                            padding=True,
                                            # truncation='longest_first',
                                            # max_length=8000
                                            ) #, use_auth_token=token, cache_dir=cache_directory
    # tokenizer = AutoTokenizer.from_pretrained(model, padding=True)

    start = time.time()

    input_path = "/n/fs/rnspace/projects/vertaix/nlp4matbench/data"
    # train = pd.read_csv(f"{input_path}/{dataset_name}/unfiltered/train.csv")
    data = pd.read_csv(f"{input_path}/{dataset_name}/unfiltered/{dataset_name}_prompting_data_chat_struct_and_descr.csv")
    # text = """<s>[INST] <<SYS>>
    # You are a material scientist. 
    # Look at the chemical formula of the given crystalline material and predict its property.
    # The output must be in a json format. For example: {property_name:predicted_property_value}.
    # Answer as precise as possible and in few words as possible.
    # <</SYS>>

    # chemical formula: KPrMnNbO6.
    # property name: band_gap. [/INST]"""

    data = data.dropna(subset=[property_name])

    prompts = list(data[f'{property_name}_{input_type}_{prompt_type}'])

    results = generate(model, tokenizer, prompts, max_len, batch_size) #prompts_all[property][0:10]

    save_path = f"{statistics_directory}{model_name}_test_stats_for_{property_name}_{input_type}_{prompt_type}_{max_len}_{batch_size}.json"
    writeToJSON(results, save_path)
    # print(results)

    end = time.time()
    print('took:', end-start)

    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model=model,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    # )

    # sequences = pipeline(
    #     'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    #     do_sample=True,
    #     top_k=10,
    #     num_return_sequences=1,
    #     eos_token_id=tokenizer.eos_token_id,
    #     max_length=200,
    # )
    # for seq in sequences:
    #     print(f"Result: {seq['generated_text']}")
