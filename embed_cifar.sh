#!/bin/bash

torchrun --nproc-per-node 8 -m open_lm.make_embeddings \
	--train-data "/fsx/iejmac/code/open_lm/open_lm/eval_set/cifar/tokens_train/part_{0..7}.tar" \
	--train-num-samples 60000 \
	--workers 6 \
	--dataset-resampled \
	--precision amp_bfloat16 \
	--model "vqgpt_vbig_4k_1024" \
	--resume "/fsx/iejmac/experiments/open_lm/vq_gpt-vqgpt_vbig_4k_1024-0.003-0.1-180000000000-2000/checkpoints/epoch_9.pt"\
	--name "vbig_180B_train" \
	--logs "embeddings/cifar" \
	--model-norm gain_only_layer_norm \
	--qk-norm \
