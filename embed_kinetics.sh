#!/bin/bash

# TODO: CHANGE THE DATA DUMBASS

# DATA="pipe:aws s3 cp s3://stability-west/kinetics400/train_tokenized_movq_256/{00000..00246}_clip_embeddings.tar -"
# GPUS=8
# SPLIT="train"

DATA="pipe:aws s3 cp s3://stability-west/kinetics400/val_tokenized_movq_256/{00000..00019}_clip_embeddings.tar -"
GPUS=5
SPLIT="test"


torchrun --nproc-per-node $GPUS -m open_lm.make_embeddings \
 	--train-data "$DATA" \
	--train-num-samples 300000 \
	--workers 6 \
	--precision amp_bfloat16 \
	--model "vqgpt_small_4k_16384" \
	--resume "/fsx/iejmac/experiments/open_lm/vq_gpt-vqgpt_small_4k_16384-0.003-0.1-10000000000-200-movq_2Mbs/checkpoints/epoch_1.pt" \
	--name "movq_hdvila_small_10B_$SPLIT" \
	--logs "embeddings/kinetics" \
	--model-norm gain_only_layer_norm \
	--qk-norm \
