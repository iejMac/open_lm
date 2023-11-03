#!/bin/bash

# 	--nps-dir "/fsx/iejmac/junk/art/dataset/dataset_tokenized" \
# 	--checkpoint "gen_checkpoints/hf_checkpoint_big_hd/" \
# 	--nps-dir "/fsx/iejmac/code/open_lm/open_lm/gen_data/8192_shard" \
# 	--out-dir "/fsx/iejmac/code/open_lm/open_lm/gen_checkpoints/hf_checkpoint_big_hd/genned_seed2_t1p5" \

python3 -m open_lm.distributed_generation \
	--checkpoint "gen_checkpoints/small_4k_16384/" \
	--nps-dir "/fsx/iejmac/code/open_lm/open_lm/gen_data/16384/kinetics" \
	--out-dir "/fsx/iejmac/code/open_lm/open_lm/gen_checkpoints/small_4k_16384/genned_kinetics" \
	--seed 0 \
	--start-idx 0 \
	--n-gen-frames 8 \
