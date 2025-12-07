#!/bin/bash

set -x

# =============================================================================
# Configuration - Fill in your paths here
# =============================================================================

# Model path: Path to your pre-trained or edited model
MODEL_PATH=""

# Data paths: Training and validation data in parquet format
TRAIN_DATA=""
VAL_DATA=""

# Tensorboard directory: Where to save training logs
TENSORBOARD_DIR=""

# Experiment name: Identifier for this training run
EXPERIMENT_NAME="my_experiment"

# Number of GPUs to use
NUM_GPUS=4

# Optional: Specify which GPUs to use (comment out to use all available)
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# =============================================================================
# Advanced Configuration (optional)
# =============================================================================

# Reward function and format template
REWARD_FUNCTION="./examples/reward_function/KE_multi_v2_rewards.py:designed_reward"
FORMAT_PROMPT="./examples/format_prompt/KE_format_cot.jinja"

# =============================================================================
# Execution
# =============================================================================

# Create tensorboard directory
mkdir -p "${TENSORBOARD_DIR}"
export TENSORBOARD_DIR
export PYTHONUNBUFFERED=1

# Print configuration
echo "=============================="
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Model: ${MODEL_PATH}"
echo "Train Data: ${TRAIN_DATA}"
echo "Val Data: ${VAL_DATA}"
echo "Tensorboard: ${TENSORBOARD_DIR}"
echo "GPUs: ${NUM_GPUS}"
echo "=============================="

# Run training
python3 -m verl.trainer.main \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    config=examples/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.format_prompt=${FORMAT_PROMPT} \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.logger="['console','tensorboard']" \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.fsdp.torch_dtype=bf16 \
    data.rollout_batch_size=512 \
    worker.actor.global_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    worker.rollout.n=10 \
    data.val_batch_size=96 \
    algorithm.adv_estimator=grpo \
    data.prompt_key=prompt \
    data.answer_key=ground_truth \
    trainer.total_epochs=1000
