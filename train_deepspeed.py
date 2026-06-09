import deepspeed
import torch

from model import get_model
from dataset import ToyDataset

def run():
    model = get_model()

    ds_config = {
        "train_batch_size": 2,
        "fp16": {"enabled": True},
        "zero_optimization": {"stage": 2}
    }

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    loader = torch.utils.data.DataLoader(ToyDataset(), batch_size=2)

    for step, batch in enumerate(loader):
        batch = {k: v.to(model_engine.device) for k, v in batch.items()}

        loss = model_engine(**batch).loss
        model_engine.backward(loss)
        model_engine.step()

        print({"step": step, "loss": loss.item()})

        if step == 50:
            break

if __name__ == "__main__":
    run()
