#!/bin/bash
#SBATCH --partition=g40
#SBATCH --job-name=vq_gpt
#SBATCH --nodes 32
#SBATCH --exclude ip-26-0-159-213,ip-26-0-158-116
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/%x_%j.out
#SBATCH --account=stability
#SBATCH --open-mode=append
#SBATCH --exclusive
module load openmpi
module load cuda/11.8

export MASTER_ADDR=`hostname`
export MASTER_PORT=12802
export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG=info

export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0

source /fsx/iejmac/code/open_lm/.env/bin/activate
cd /fsx/iejmac/code/open_lm

LR=0.003

# TOKENS=180000000000
TOKENS=66000000000
# TOKENS=10000000000
SAVES=1

# TOKENS=60000000000
# SAVES=3

BATCHSIZE=2 # (total BS should be 4M)

# WARM=2000
WARM=1000
# WARM=200

MODEL="vqgpt_big_8k_8192"
WD=0.1
ACC=1 # 4, 8

TOTAL_TOKENS=`expr $TOKENS \* $SAVES`
IM_W=0.3
VID_W=0.7
VARIANT="$IM_W-$VID_W-mix"

EXP_NAME="vq_gpt-$MODEL-$LR-$WD-$TOTAL_TOKENS-$WARM-$VARIANT"

echo "node-list: $SLURM_JOB_NODELIST"

srun --cpu_bind=v --accel-bind=gn python3 -m open_lm.main \
    --train-data "pipe:aws s3 cp s3://stability-west/laion_coyo_vqgan_tokenized_open_lm2/tars-8200/shard-{0000000..0000482}.tar -"  "pipe:aws s3 cp s3://stability-west/stable-video-dataset-tokenized/open_lm_test2/tars-8200/shard-{0000000..0001723}.tar -" \
    --train-data-mix-weights $IM_W $VID_W \
    --workers 2 \
    --train-num-samples $TOKENS \
    --precision amp_bfloat16 \
    --batch-size $BATCHSIZE \
    --grad-checkpointing \
    --log-every-n-steps 20 \
    --grad-clip-norm 1 \
    --lr $LR \
    --warmup $WARM \
    --model $MODEL \
    --wd $WD \
    --beta2 0.95 \
    --epochs $SAVES \
    --report-to wandb \
    --wandb-project-name vq_gpt \
    --name $EXP_NAME \
    --logs /fsx/iejmac/experiments/open_lm \
    --resume latest \
    --seed 42 \
    --accum-freq $ACC \
    --model-norm gain_only_layer_norm \
    --qk-norm \
    --data-key 'txt' \
    # --fsdp \
    # --fsdp-amp \
    # --delete-previous-checkpoint
    # --load-pretrained-state \
    # --lr-cooldown-end 3e-5
