import argparse
import logging
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from model import load_tokenizer_and_model, GPT2Dataset, tokenize_data
import pandas as pd
import re
from functools import partial
from sklearn.model_selection import train_test_split
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_args():
    parser = argparse.ArgumentParser(description="Train and validate GPT-2 on a custom dataset")
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for the optimizer')
    parser.add_argument('--model_name', type=str, default='gpt2', help='Model identifier for the pretrained model to use')
    parser.add_argument('--dataset_path', type=str, default='../data/bad_reviews_dataset.csv', help='File path to the dataset')
    return parser.parse_args()

def load_dataset(filepath):
    """
    Load dataset and prepare for tokenization. Replace special characters from Review and Suggestion fields.
    """
    try:
        df = pd.read_csv(filepath)
        df['Review'] = df['Review'].str.replace(r"[^a-zA-Z\s]", "", regex=True)
        df['Improvement_Suggestion'] = df['Improvement_Suggestion'].str.replace(r"[^a-zA-Z\s]", "", regex=True)
        
        df['Input'] = df.apply(lambda row: 
            f"[{row['Sentiment'].upper()}] {row['Review']} Suggestion: {row['Improvement_Suggestion']}", 
            axis=1
        )
        
        train_texts, val_texts = train_test_split(df['Input'].tolist(), test_size=0.2, random_state=42)
        return train_texts, val_texts
    except Exception as e:
        logging.error(f"Error processing dataset: {str(e)}")
        return [], []

