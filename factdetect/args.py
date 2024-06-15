
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Argument parser for factdetect')
    
    # Add your command-line arguments here
    parser.add_argument('--corpus', help='Path to corpus file',
                        default='../multivers/data_train/target_af/scifact_10/corpus.jsonl')
    parser.add_argument('--test_file', help='Path to input test file',
                        default='../multivers/data_train/target_af/scifact_10/claims_dev_qa2c_nei_res.jsonl')
    parser.add_argument('--mode', help='train or eval',default='train')
    parser.add_argument('--llm_model_name', help='model name')
    parser.add_argument('--outfilename', help='output file path',
                        default='predictions.jsonl')
    parser.add_argument('--af_mode', help='af mode',action='store_true')
    parser.add_argument('--prompt_type', help='prompt type',default='factdetect')
    #add preds_file 
    parser.add_argument('--preds_file', help='Path to input test file',
                        default='predictions.jsonl')
    parser.add_argument('--with_nei',action='store_true')
    parser.add_argument('--factuality',action='store_true')
    
    return parser.parse_args()


