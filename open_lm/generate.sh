#!/bin/bash

python3 -m open_lm.distributed_generation \
	--checkpoint "hf_checkpoint_vbig/" \
	--nps-dir "/fsx/iejmac/junk/art/dataset/dataset_tokenized" \
	--out-dir "vbig_66B_test" \
	--start-idx 10 \
	--n-gen-frames 16 \
