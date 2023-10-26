#!/bin/bash

torchrun --nproc-per-node 8 -m open_lm.make_embeddings \
	--train-data "/fsx/iejmac/code/open_lm/open_lm/eval_set/cifar/tokens_train_8k/part_{0..7}.tar" \
	--train-num-samples 60000 \
	--workers 6 \
	--dataset-resampled \
	--precision amp_bfloat16 \
	--model "vqgpt_big_4k_8192" \
	--resume "/fsx/iejmac/experiments/open_lm/vq_gpt-vqgpt_big_4k_8192-0.003-0.1-66000000000-1000/checkpoints/epoch_1.pt" \
	--name "big_66B_8k_train" \
	--logs "embeddings/cifar" \
	--model-norm gain_only_layer_norm \
	--qk-norm \
