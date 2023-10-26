#!/bin/bash

torchrun --nproc-per-node 8 -m open_lm.make_embeddings \
	--train-data "/fsx/iejmac/dataset/ssv2/test_tokens/{00000..00007}_clip_embeddings.tar" \
	--train-num-samples 170000 \
	--workers 6 \
	--dataset-resampled \
	--precision amp_bfloat16 \
	--model "vqgpt_small_4k_1024" \
	--resume "/fsx/iejmac/experiments/open_lm/vq_gpt-vqgpt_small_4k_1024-2-0.003-0.1-2-60000000000-2000/checkpoints/epoch_3.pt" \
	--name "small_180B_test" \
	--logs "embeddings/ssv2" \
	--model-norm gain_only_layer_norm \
	--qk-norm \
