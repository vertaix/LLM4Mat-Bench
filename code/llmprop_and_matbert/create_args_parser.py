import argparse

def args_parser():

    parser = argparse.ArgumentParser(description='LLM4Mat-Bench')
    
    parser.add_argument('--epochs',
                        help='Number of epochs',
                        type=int,
                        default=200)
    parser.add_argument('--train_bs',
                        help='Batch size',
                        type=int,
                        default=64)
    parser.add_argument('--inference_bs',
                        help='Batch size',
                        type=int,
                        default=1024)
    parser.add_argument('--lr',
                        help='Learning rate',
                        type=float,
                        default=0.001)
    parser.add_argument('--max_len',
                        help='Max input sequence length',
                        type=int,
                        default=888)
    parser.add_argument('--dr',
                        help='Drop rate',
                        type=float,
                        default=0.5)
    parser.add_argument('--warmup_steps',
                        help='Warmpup steps',
                        type=int,
                        default=30000)
    parser.add_argument('--preprocessing_strategy',
                        help='Data preprocessing technique: "none", "bond_lengths_replaced_with_num", "bond_angles_replaced_with_ang", "no_stopwords", or "no_stopwords_and_lengths_and_angles_replaced"',
                        type=str,
                        default="no_stopwords_and_lengths_and_angles_replaced")
    parser.add_argument('--tokenizer',
                        help='Tokenizer name: "t5_tokenizer", "llmprop_tokenizer", or "matbert_tokenizer"',
                        type=str,
                        default="llmprop_tokenizer")
    parser.add_argument('--pooling', 
                        help='Pooling method. "cls" or "mean"',
                        type=str,
                        default="cls")
    parser.add_argument('--normalizer', 
                        help='Labels scaling technique. "z_norm", "mm_norm", or "ls_norm"',
                        type=str,
                        default="z_norm") 
    parser.add_argument('--scheduler', 
                        help='Learning rate scheduling technique. "linear", "onecycle", "step", or "lambda" (no scheduling))',
                        type=str,
                        default="onecycle")
    parser.add_argument('--property_name', 
                        help='The name of the property to predict. "band_gap", "volume", or "is_gap_direct"',
                        type=str,
                        default="band_gap")
    parser.add_argument('--optimizer', 
                        help='Optimizer type. "adamw" or "sgd"',
                        type=str,
                        default="adamw")
    parser.add_argument('--task_name', 
                        help='the name of the task: "regression" if propert_name is band_gap or volume or "classification" if property_name is is_gap_direct',
                        type=str,
                        default="regression")
    parser.add_argument('--data_path',
                        help="the path to the data",
                        type=str,
                        default="data/")                    
    parser.add_argument('--checkpoints_path',
                        help="the path to the best checkpoint for evaluation",
                        type=str,
                        default="checkpoints/")
    parser.add_argument('--results_path',
                        help="the path to the directory where results are saved",
                        type=str,
                        default="results/")
    parser.add_argument('--tokenizers_path',
                        help="the path to the directory where tokenizers are saved",
                        type=str,
                        default="tokenizers/") 
    parser.add_argument('--input_type',
                        help="description, structure, or composition",
                        type=str,
                        default="description")
    parser.add_argument('--dataset_name',
                        help="mp, ...",
                        type=str,
                        default="mp")
    parser.add_argument('--model_name',
                        help="llmprop or matbert",
                        type=str,
                        default="llmprop")
    parser.add_argument('--regressor',
                        help="linear, ...",
                        type=str,
                        default="linear")
    parser.add_argument('--loss',
                        help="mae, ...",
                        type=str,
                        default="mae")
    args = parser.parse_args()
    
    return args
