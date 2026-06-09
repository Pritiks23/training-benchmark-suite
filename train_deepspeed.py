import json
import torch
import deepspeed

from model import get_model
from dataset import ToyDataset
from metrics import gpu_util, mem


# -----------------------------
# DeepSpeed config (minimal stable)
# -----------------------------
ds_config = {
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 1,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2
    }
}


def run():

    # -----------------------------
    # Model
    # -----------------------------
    model = get_model()
    model.train()

    # REQUIRED for DeepSpeed ZeRO
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

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

    # -----------------------------
    # Training loop
    # -----------------------------
    for step, batch in enumerate(loader):

        # move batch to GPU
        batch = {k: v.to(model_engine.device) for k, v in batch.items()}

        # -----------------------------
        # SAFE forward (THIS FIXES YOUR ERROR)
        # -----------------------------
        outputs = model_engine(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            return_dict=True
        )

        loss = outputs.loss

        # -----------------------------
        # backward + step
        # -----------------------------
        model_engine.backward(loss)
        model_engine.step()

        # -----------------------------
        # logging
        # -----------------------------
        print(json.dumps({
            "step": step,
            "gpu": gpu_util(),
            "mem": mem(),
            "loss": float(loss.item())
        }))

        if step >= 50:
            break


if __name__ == "__main__":
    run()
