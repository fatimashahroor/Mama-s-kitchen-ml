import pandas as pd
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_tokenizer_and_model(model_name='gpt2'):
    """
    Load the tokenizer and model from the Hugging Face library.
    Ensure the tokenizer has a distinct pad token.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set pad token to <|pad|> if not already set
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model

