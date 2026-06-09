import torch
import json
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from model import get_model
from dataset import ToyDataset
from metrics import Timer, gpu_util, mem


def run():
    # 🔥 REQUIRED for FSDP
    dist.init_process_group(backend="nccl", init_method="env://")

    # Get device from distributed rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Model
    model = get_model().cuda()
    model = FSDP(model)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Data
    loader = DataLoader(ToyDataset(), batch_size=2)

    for step, batch in enumerate(loader):
        batch = {k: v.cuda() for k, v in batch.items()}

        with Timer() as t:
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            opt.zero_grad()

        log = {
            "step": step,
            "time_ms": t.dt,
            "gpu": gpu_util(),
            "mem": mem(),
            "loss": loss.item()
        }

        print(json.dumps(log))

        if step >= 50:
            break

    dist.destroy_process_group()


if __name__ == "__main__":
    import os
    run()
