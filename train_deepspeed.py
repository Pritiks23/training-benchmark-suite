import torch
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM

# HARD LOCAL PATH (no HF hub calls allowed)
MODEL_PATH = "/workspace/gpt2_local"


def run():

    # -------------------------
    # FORCE OFFLINE TOKENIZER
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    use_fast=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.float16
    )

    model.train()
    model.cuda()

    # -------------------------
    # DUMMY INPUT (replace later with dataset)
    # -------------------------
    text = "DeepSpeed training test input"
    inputs = tokenizer(text, return_tensors="pt")

    inputs = {k: v.cuda() for k, v in inputs.items()}

    # -------------------------
    # MINIMAL DEEPSPEED CONFIG
    # -------------------------
    ds_config = {
        "train_batch_size": 1,
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 0
        }
    }

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    # -------------------------
    # FORWARD + BACKWARD STEP
    # -------------------------
    outputs = model_engine(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss

    model_engine.backward(loss)
    model_engine.step()

    print("SUCCESS: DeepSpeed step complete")
    print("Loss:", loss.item())


if __name__ == "__main__":
    run()
