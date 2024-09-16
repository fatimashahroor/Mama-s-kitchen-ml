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

def load_and_tokenize_data(filepath, tokenizer):
    """
    Load data from CSV and tokenize using the provided tokenizer.
    Combines 'Sentiment', 'Review', and 'Improvement_Suggestion'.
    """
    df = pd.read_csv(filepath)
    
    df['Input'] = df.apply(
        lambda row: f"[{row['Sentiment'].upper()}] {row['Review']} Suggestion: {row['Improvement_Suggestion']}", 
        axis=1
    )
    
    df = df.dropna(subset=['Input'])
    inputs = df['Input'].tolist()

    tokenized_data = tokenizer(
        inputs, 
        max_length=512, 
        truncation=True, 
        padding="max_length", 
        return_tensors="pt"
    )
    
    return tokenized_data
