#!/bin/bash

echo "Running PyTorch baseline..."
python train_torch.py > results/torch.log

echo "Running FSDP..."
python train_fsdp.py > results/fsdp.log

echo "Running DeepSpeed..."
python train_deepspeed.py > results/deepspeed.log
