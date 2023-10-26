#!/bin/bash

# 	--nps-dir "/fsx/iejmac/junk/art/dataset/dataset_tokenized" \
# 	--checkpoint "gen_checkpoints/hf_checkpoint_big_hd/" \
# 	--nps-dir "/fsx/iejmac/code/open_lm/open_lm/gen_data/8192_shard" \
# 	--out-dir "/fsx/iejmac/code/open_lm/open_lm/gen_checkpoints/hf_checkpoint_big_hd/genned_seed2_t1p5" \

python3 -m open_lm.distributed_generation \
	--checkpoint "gen_checkpoints/hf_checkpoint_big_ld/" \
	--nps-dir "/fsx/iejmac/code/open_lm/open_lm/gen_data/1024_shard" \
	--out-dir "/fsx/iejmac/code/open_lm/open_lm/gen_checkpoints/hf_checkpoint_big_ld/genned_seed0_t1p5" \
	--seed 0 \
	--start-idx 0 \
	--n-gen-frames 7 \
