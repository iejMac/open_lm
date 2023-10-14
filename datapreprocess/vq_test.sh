#!/bin/bash

python3 make_vq.py \
	--input-files "s3://stability-west/acav/tokenized_vqgan_8192_8x/" \
	--output-dir "tokens" \
	--num-workers 16 \
	--num-consumers 32 \
	--upload-to-s3 \
	--s3-path "s3://stability-west/acav/open_lm_tokens_8192/" \
	--vocab-size 8192 \
	--tokens-per-frame 1024 \
	--n-frames 4 \
