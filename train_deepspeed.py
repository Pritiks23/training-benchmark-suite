import os
import json
import torch
import deepspeed

from model import get_model
from dataset import ToyDataset
from metrics import Timer, gpu_util, mem


def run():

    # -----------------------------
    # Model
    # -----------------------------
    model = get_model()

    # -----------------------------
    # Optimizer (REQUIRED for ZeRO)
    # -----------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # -----------------------------
    # DeepSpeed config
    # -----------------------------
    ds_config = {
        "train_batch_size": 2,
        "train_micro_batch_size_per_gpu": 2,
        "zero_optimization": {
            "stage": 2
        },
        "fp16": {
            "enabled": True
        }
    }

    # -----------------------------
    # Init DeepSpeed engine
    # -----------------------------
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config
    )

    # -----------------------------
    # Data
    # -----------------------------
    loader = ToyDataset()

    for step, batch in enumerate(loader):

        # Move tensors to GPU
        batch = {k: v.to(model_engine.device) for k, v in batch.items()}

        # -----------------------------
        # Forward (SAFE HF FORMAT)
        # -----------------------------
        outputs = model_engine(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
            labels=batch["labels"]
        )

        loss = outputs.loss

        # -----------------------------
        # Backward + step
        # -----------------------------
        model_engine.backward(loss)
        model_engine.step()

        # -----------------------------
        # Logging (benchmark format)
        # -----------------------------
        print(json.dumps({
            "step": step,
            "time_ms": None,  # optional if you add Timer later
            "gpu": gpu_util(),
            "mem": mem(),
            "loss": loss.item()
        }))

        if step >= 50:
            break


if __name__ == "__main__":
    run()
