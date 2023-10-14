from open_lm.utils.transformers.model import OpenLMforCausalLM
from transformers import GPTNeoXTokenizerFast
import argparse
import torch
import glob
import numpy as np
import torch.distributed as dist
from precision import get_autocast

def main(rank, world_size, args):
    n_ctx_tokens = (257 * 16)
    n_gen_frames = args.n_gen_frames
    
    nps = glob.glob(f"{args.nps_dir}/*.npy")
    
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

        vid = x[args.start_idx:args.start_idx+15]
        vid = np.hstack([vid, np.full((vid.shape[0], 1), 1024)])
        vid = vid.reshape(-1)
        vid = vid[None, :n_ctx_tokens]

        init_ids = torch.from_numpy(vid).long().to(rank)
        input_ids = torch.clone(init_ids)
        frames = []

        with torch.no_grad(), autocast():
            for f in range(n_gen_frames):
                greedy_output = model.generate(input_ids, max_length=n_ctx_tokens, do_sample=True, top_p=0.9)
                out_np = greedy_output.cpu().numpy()

                frames.append(out_np[:, -(257):])
                input_ids = greedy_output[:, 257:]

        out_vid = np.concatenate([vid.reshape(-1, 257)] + frames)

        print(f"Saving {args.out_dir}/vid{start_idx + i}.npy of shape {out_vid.shape}")
        np.save(f"{args.out_dir}/vid{start_idx + i}.npy", out_vid)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--nps-dir", type=str, required=True, help="Directory containing .npy files to process.")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to save the generated samples.")
    parser.add_argument("--start-idx", type=int, default=0, required=False, help="Starting frame index of each video.")
    parser.add_argument("--n-gen-frames", type=int, default=8, required=False, help="Starting frame index of each video.")
    args = parser.parse_args()

    world_size = 8
    # Spawn multiple processes for each GPU
    torch.multiprocessing.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
