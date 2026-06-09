from transformers import GPT2LMHeadModel

def get_model():
    return GPT2LMHeadModel.from_pretrained("gpt2")
