#!/bin/bash

# 	--train-data "/fsx/iejmac/code/open_lm/open_lm/eval_set/cifar/tokens_test/part_{0..7}.tar" \
# 	--train-data-upsampling-factors "/fsx/iejmac/code/open_lm/open_lm/eval_set/cifar/tokens_train/classes15.npy" \
torchrun --nproc-per-node 8 -m open_lm.distributed_fs \
	--train-data "/fsx/iejmac/code/open_lm/open_lm/eval_set/cifar/tokens_test_16k/part_{0..7}.tar" \
	--train-num-samples 60000 \
	--workers 6 \
	--dataset-resampled \
	--precision amp_bfloat16 \
	--model "vqgpt_big_16k" \
	--resume "/fsx/iejmac/experiments/open_lm/vq_gpt-vqgpt_big_16k-2-0.002-0.1-2-14000000000-1000/checkpoints/epoch_1.pt"\
	--train-data-upsampling-factors "/fsx/iejmac/code/open_lm/open_lm/eval_set/cifar/tokens_train_16k/classes15.npy" \
	--model-norm gain_only_layer_norm \
	--qk-norm \
