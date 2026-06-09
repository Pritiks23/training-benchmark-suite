import time
import json
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "gpt2"  # keep stable for benchmark consistency

def gpu_util():
    return torch.cuda.utilization() if torch.cuda.is_available() else 0

def mem():
    return torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0


def make_batch(tokenizer):
    text = "Hello world " * 32
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # IMPORTANT: GPT2 sometimes has no attention_mask dependency in older wrappers
    return {
        "input_ids": tokens["input_ids"],
        "labels": tokens["input_ids"].clone(),
    }


def run():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config={
            "train_batch_size": 1,
            "fp16": {"enabled": False},
            "zero_optimization": {"stage": 2},
        },
    )

    batch = make_batch(tokenizer)

    for step in range(50):
        start = time.time()

        outputs = model_engine(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
        )

        loss = outputs.loss
        model_engine.backward(loss)
        model_engine.step()

        dt = (time.time() - start) * 1000

        print(json.dumps({
            "step": step,
            "time_ms": dt,
            "gpu": gpu_util(),
            "mem": mem(),
            "loss": loss.item()
        }))


if __name__ == "__main__":
    run()
