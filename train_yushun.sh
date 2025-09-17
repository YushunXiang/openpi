#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_TRACEBACK_FILTERING=off

train_prefix="pi05_pour_water"
train_name="${train_prefix}_$(date +%Y%m%d_%H%M%S)"
uv run scripts/train.py "$train_prefix" --exp-name="$train_name" --overwrite
