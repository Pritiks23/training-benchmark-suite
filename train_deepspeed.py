import deepspeed
import torch
import json

from model import get_model
from dataset import ToyDataset


def run():

    model = get_model().cuda()

    # ✅ MUST define optimizer BEFORE deepspeed.init
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ⚠️ DeepSpeed config (auto or dict)
    ds_config = {
        "train_batch_size": 2,
        "zero_optimization": {
            "stage": 2
        },
        "fp16": {
            "enabled": True
        }
    }

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config
    )

    loader = ToyDataset()

    for step, batch in enumerate(loader):

        batch = {k: v.cuda() for k, v in batch.items()}

        loss = model_engine(**batch).loss
        model_engine.backward(loss)
        model_engine.step()

        print(json.dumps({
            "step": step,
            "loss": loss.item()
        }))

        if step >= 50:
            break


if __name__ == "__main__":
    run()
