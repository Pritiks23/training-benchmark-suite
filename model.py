from transformers import GPT2LMHeadModel

def get_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return model
