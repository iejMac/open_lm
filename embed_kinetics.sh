#!/bin/bash

#  	--train-data "pipe:aws s3 cp s3://stability-west/kinetics400/train_tokenized_vqgan_1024_16x/{00000..00246}_clip_embeddings.tar -" \
# 	--train-num-samples 16000 \
torchrun --nproc-per-node 5 -m open_lm.make_embeddings \
 	--train-data "pipe:aws s3 cp s3://stability-west/kinetics400/val_tokenized_vqgan_1024_16x/{00000..00019}_clip_embeddings.tar -" \
	--train-num-samples 300000 \
	--workers 6 \
	--precision amp_bfloat16 \
	--model "vqgpt_small_4k_1024" \
	--resume "/fsx/iejmac/experiments/open_lm/vq_gpt-vqgpt_small_4k_1024-2-0.003-0.1-2-60000000000-2000/checkpoints/epoch_3.pt" \
	--name "small_180B_test" \
	--logs "embeddings/kinetics" \
	--model-norm gain_only_layer_norm \
	--qk-norm \
