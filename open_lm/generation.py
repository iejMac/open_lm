from utils.transformers.model import OpenLMforCausalLM
from transformers import GPTNeoXTokenizerFast
import argparse 
import torch
import glob
import numpy as np

from precision import get_autocast

if __name__ == '__main__':

    # n_ctx_tokens = (257 * (16 - 8))
    n_ctx_tokens = (257 * 16)
    n_gen_frames = 16

    nps = glob.glob("/fsx/iejmac/taming-transformers/test/*.npy")
    out_dir = "small_180B_test"

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    # parser.add_argument('--prompt', type=str, default='i')
    args = parser.parse_args()
    model = OpenLMforCausalLM.from_pretrained(args.checkpoint)
    model = model.cuda()

    autocast = get_autocast("amp_bfloat16")

    for i, arr_path in enumerate(nps):
        x = np.load(arr_path)

        vid = x[:15]
        vid = np.hstack([vid, np.full((vid.shape[0], 1), 1024)])
        vid = vid.reshape(-1)
        vid = vid[None, :n_ctx_tokens]

        init_ids = torch.from_numpy(vid).long()
        input_ids = torch.clone(init_ids)
        frames = []

        with torch.no_grad(), autocast():
            for f in range(n_gen_frames):
                greedy_output = model.generate(input_ids.to(0), max_length=n_ctx_tokens, do_sample=True, top_p=0.9)
                out_np = greedy_output.cpu().numpy()

                frames.append(out_np[:, -(257):])
                input_ids = greedy_output[:, 257:]

        # out_vid = out_np
        out_vid = np.concatenate([vid.reshape(-1, 257)] + frames)
        print((out_vid == 1024).nonzero())
        
        np.save(f"{out_dir}/vid{i}.npy", out_vid)
