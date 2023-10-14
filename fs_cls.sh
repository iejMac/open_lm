#!/bin/bash

torchrun --nproc-per-node 8 -m open_lm.distributed_fs \
	--train-data "/fsx/iejmac/code/open_lm/open_lm/eval_set/cifar/tokens_test/part_{0..7}.tar" \
	--train-num-samples 60000 \
	--workers 6 \
	--dataset-resampled \
	--precision amp_bfloat16 \
	--model "vqgpt_vbig_4k_1024" \
	--resume "/fsx/iejmac/experiments/open_lm/vq_gpt-vqgpt_vbig_4k_1024-2-0.003-0.1-2-10000000000-200/checkpoints/epoch_1.pt"\
	--train-data-upsampling-factors "/fsx/iejmac/code/open_lm/open_lm/eval_set/cifar/tokens_train/classes15.npy" \
	--model-norm gain_only_layer_norm \
	--qk-norm \
