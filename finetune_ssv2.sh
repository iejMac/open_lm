#!/bin/bash

torchrun --nproc-per-node 8 -m open_lm.finetune \
	--train-data "/fsx/iejmac/dataset/ssv2/train_tokens/{00000..00007}_clip_embeddings.tar" \
	--train-num-samples 200000 \
	--val-data "/fsx/iejmac/dataset/ssv2/test_tokens/{00000..00007}_clip_embeddings.tar" \
	--workers 6 \
	--dataset-resampled \
	--precision amp_bfloat16 \
	--lr 0.07 \
	--accum-freq 4 \
	--model "vqgpt_small_4k_1024" \
	--resume "/fsx/iejmac/experiments/open_lm/vq_gpt-vqgpt_small_4k_1024-4-0.003-0.1-4-10000000000-200/checkpoints/epoch_1.pt" \
	--model-norm gain_only_layer_norm \
	--qk-norm \
