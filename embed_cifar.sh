#!/bin/bash

torchrun --nproc-per-node 8 -m open_lm.make_embeddings \
	--train-data "/fsx/iejmac/code/open_lm/open_lm/eval_set/cifar/tokens_test_8k/part_{0..7}.tar" \
	--train-num-samples 60000 \
	--workers 6 \
	--precision amp_bfloat16 \
	--model "vqgpt_small_8k_8192" \
	--resume "/fsx/iejmac/experiments/open_lm/vq_gpt-vqgpt_small_8k_8192-0.003-0.1-66000000000-1000-0.5-0.5-mix/checkpoints/epoch_1.pt" \
	--name "small_66B_0p5_im_test" \
	--logs "embeddings/cifar/8k_vocab" \
	--model-norm gain_only_layer_norm \
	--qk-norm \
