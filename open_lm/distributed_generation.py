import os
from transformers import GPTNeoXTokenizerFast, set_seed
import argparse
import torch
import glob
import numpy as np
import torch.distributed as dist
from precision import get_autocast

from open_lm.utils.transformers.hf_model import OpenLMforCausalLM


# TOKENS_PER_FRAME, VOCAB_SIZE, N_CTX_FRAMES = 256, 1024, 16
TOKENS_PER_FRAME, VOCAB_SIZE, N_CTX_FRAMES = 256, 16384, 16
# TOKENS_PER_FRAME, VOCAB_SIZE, N_CTX_FRAMES = 1024, 8192, 8
MAX_VIDS=128


def main(rank, world_size, args):
    n_ctx_tokens = ((TOKENS_PER_FRAME+1) * N_CTX_FRAMES)
    n_gen_frames = args.n_gen_frames

    set_seed(args.seed)
    
    nps = glob.glob(f"{args.nps_dir}/*.npy")[:MAX_VIDS]
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Splitting the data for each process
    nps_per_gpu = len(nps) // world_size
    start_idx = rank * nps_per_gpu
    end_idx = start_idx + nps_per_gpu if rank != world_size - 1 else len(nps)
    my_nps = nps[start_idx:end_idx]

    model = OpenLMforCausalLM.from_pretrained(args.checkpoint)
    model = model.cuda(rank) # Use the specific GPU for this process

    autocast = get_autocast("amp_bfloat16")

    for i, arr_path in enumerate(my_nps):
        x = np.load(arr_path)

        vid = x[args.start_idx:args.start_idx+N_CTX_FRAMES-1]
        vid = np.hstack([vid, np.full((vid.shape[0], 1), VOCAB_SIZE)])
        vid = vid.reshape(-1)
        vid = vid[None, :n_ctx_tokens]

        init_ids = torch.from_numpy(vid).long().to(rank)
        input_ids = torch.clone(init_ids)
        frames = []

        with torch.no_grad(), autocast():
            f = 0
            while f < n_gen_frames:
                greedy_output = model.generate(input_ids, max_length=n_ctx_tokens, do_sample=True, temperature=1.0, top_p=0.95)

                out_np = greedy_output.cpu().numpy()
                out_np[:, -1] = VOCAB_SIZE # Ensure delimiters are correct

                last_input_frame = out_np[:, -2*(TOKENS_PER_FRAME+1):-(TOKENS_PER_FRAME+1)]
                genned_frame = out_np[:, -(TOKENS_PER_FRAME+1):]

                overlap = (genned_frame[:, :-1] == last_input_frame[:, :-1]).sum(axis=-1)[0] / TOKENS_PER_FRAME

                '''
                if overlap < 0.01 or overlap > 0.5:
                    print(f"Overlap {overlap}, retrying...")
                    # TODO: make this optionsl
                    # TODO: add retry counter. If > 3 go to next vid
                    # can do this by setting a retry count at the start of each gen
                    # iterating here
                    # resetting to 0 if passes
                    # breaking if exceeds threshold
                    # continue in outer loop of retry_count == threshold
                    continue  # try again, either no motion or cut
                '''

                frames.append(genned_frame)
                input_ids = greedy_output[:, TOKENS_PER_FRAME+1:]
                f += 1

        out_vid = np.concatenate([vid.reshape(-1, TOKENS_PER_FRAME+1)] + frames)

        print(f"Saving {args.out_dir}/vid{start_idx + i}.npy of shape {out_vid.shape}")
        np.save(f"{args.out_dir}/vid{start_idx + i}.npy", out_vid)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--nps-dir", type=str, required=True, help="Directory containing .npy files to process.")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to save the generated samples.")
    parser.add_argument("--start-idx", type=int, default=0, required=False, help="Starting frame index of each video.")
    parser.add_argument("--n-gen-frames", type=int, default=8, required=False, help="Starting frame index of each video.")
    parser.add_argument("--seed", type=int, default=0, required=False, help="Seed for sampling")
    args = parser.parse_args()

    world_size = 8
    # Spawn multiple processes for each GPU
    torch.multiprocessing.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
