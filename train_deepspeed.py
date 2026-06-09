import torch
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = "gpt2"


def run():

    # -------------------------
    # LOAD TOKENIZER + MODEL
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    model.train()
    model.cuda()

    # -------------------------
    # DUMMY INPUT (replace with dataset later)
    # -------------------------
    text = "DeepSpeed training test input"
    inputs = tokenizer(text, return_tensors="pt")

    inputs = {k: v.cuda() for k, v in inputs.items()}

    # labels = input_ids (standard LM training)
    inputs["labels"] = inputs["input_ids"].clone()

    # -------------------------
    # FORWARD PASS (baseline sanity step)
    # -------------------------
    outputs = model(**inputs)
    loss = outputs.loss

    # -------------------------
    # DEEPSPEED CONFIG
    # -------------------------
    ds_config = {
        "train_batch_size": 1,
        "train_micro_batch_size_per_gpu": 1,
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 1
        }
    }

    # -------------------------
    # INIT DEEPSPEED
    # -------------------------
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    # -------------------------
    # BACKWARD + STEP
    # -------------------------
    model_engine.backward(loss)
    model_engine.step()

    print("✅ DeepSpeed training step completed successfully")


if __name__ == "__main__":
    run()
