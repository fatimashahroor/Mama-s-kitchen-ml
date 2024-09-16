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
def tokenize_data(texts, tokenizer, max_length=512):
    """
    Tokenizes a list of texts using the provided tokenizer.
    Args:
        texts (list): A list of text inputs.
        tokenizer: The tokenizer object from Hugging Face transformers.
        max_length (int): Maximum sequence length for padding/truncation.
    Returns:
        dict: Tokenized inputs as tensors.
    """
    return tokenizer(
        texts, 
        truncation=True, 
        padding="max_length", 
        max_length=max_length, 
        return_tensors='pt'
    )
from torch.utils.data import Dataset

class GPT2Dataset(Dataset):
    """
    GPT2 Dataset for handling the tokenization and encoding of text data.
    """
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        """
        Returns the input_ids and attention_mask at index `idx`.
        """
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
        }
