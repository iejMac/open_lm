import argparse
from utils.transformers.model import OpenLMModel
from transformers import GPTNeoXTokenizerFast
from utils.transformers.config import OpenLMConfig
from torch.nn import LayerNorm
import torch
from functools import partial
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    parser.add_argument('--model-config')
    parser.add_argument('--out-dir')
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint)
    with open(args.model_config, 'r') as f:
        config = json.load(f)

    openlm_config = OpenLMConfig(**config)

    open_lm = OpenLMModel(openlm_config)
    # hardcoded to NeoX Tokenizer
    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    state_dict = checkpoint["state_dict"]
    state_dict = {x.replace("module.",""):y for x,y in state_dict.items()}
    open_lm.model.load_state_dict(state_dict)
    open_lm.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
