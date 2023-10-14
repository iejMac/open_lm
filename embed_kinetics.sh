#!/bin/bash

#  	--train-data "pipe:aws s3 cp s3://stability-west/kinetics400/train_tokenized_vqgan_1024_16x/{00000..00246}_clip_embeddings.tar -" \
# 	--train-num-samples 16000 \
torchrun --nproc-per-node 5 -m open_lm.make_embeddings \
	--train-data "pipe:aws s3 cp s3://stability-west/kinetics400/val_tokenized_vqgan_1024_16x/{00000..00019}_clip_embeddings.tar -" \
	--train-num-samples 20000 \
	--workers 6 \
	--precision amp_bfloat16 \
	--model "vqgpt_small_4k_1024" \
	--resume "/fsx/iejmac/experiments/open_lm/vq_gpt-vqgpt_small_4k_1024-0.003-0.1-10000000000-200-filt/checkpoints/epoch_1.pt" \
	--name "small_10B_motion_test" \
	--logs "embeddings/kinetics" \
	--model-norm gain_only_layer_norm \
	--qk-norm \
