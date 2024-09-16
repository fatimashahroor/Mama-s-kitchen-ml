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

def collate_batch(batch, tokenizer):
    """
    Create a batch for training by padding input_ids and attention_mask.
    """
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

def train(model, train_loader, optimizer, device, max_grad_norm=1.0):
    """
    Train the model on the provided dataset and return average loss.
    """
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, labels = batch['input_ids'].to(device), batch['input_ids'].to(device)
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    average_loss = total_loss / len(train_loader)
    return average_loss

def validate(model, val_loader, device):
    """
    Validation function to evaluate model on a validation set.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch['input_ids'].to(device), batch['input_ids'].to(device)
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(val_loader)

def compute_perplexity(loss):
    """
    Compute perplexity based on the model's loss.
    """
    return math.exp(loss)

def generate_text(model, tokenizer, base_prompt, user_input="", max_length=100):
    """
    Generate text using the trained model based on a base prompt followed by user input.
    """
    full_prompt = f"{base_prompt} {user_input}" if user_input else base_prompt
    encoded_input = tokenizer.encode_plus(
        full_prompt,
        return_tensors='pt',
        max_length=max_length,
        padding="max_length",  
        truncation=True      
    )
    input_ids = encoded_input['input_ids'].to(model.device)
    attention_mask = encoded_input['attention_mask'].to(model.device)
    
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length + len(input_ids[0]),
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.9, 
        top_p=0.92,      
        top_k=50,       
        do_sample=True,              
        pad_token_id=tokenizer.pad_token_id  
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    processed_text = post_process_generated_text(generated_text)
    return processed_text

def post_process_generated_text(text):
    """
    Post-process the generated text to clean unwanted characters.
    """
    text = re.sub(r'[^a-zA-Z,.!? ]+', '', text)
    return text.strip()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Loading tokenizer and model...")
    tokenizer, model = load_tokenizer_and_model(args.model_name)
    model.to(device)

    logging.info("Loading dataset...")
    train_texts, val_texts = load_dataset(args.dataset_path)
    if not train_texts:
        return

    train_encodings = tokenize_data(train_texts, tokenizer)
    val_encodings = tokenize_data(val_texts, tokenizer)

    train_dataset = GPT2Dataset(train_encodings)
    val_dataset = GPT2Dataset(val_encodings)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=partial(collate_batch, tokenizer=tokenizer))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=partial(collate_batch, tokenizer=tokenizer))

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)  # Learning rate scheduler

    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    logging.info("Starting training...")
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        val_perplexity = compute_perplexity(val_loss)
        logging.info(f'Epoch {epoch}: Train Loss: {train_loss}, Validation Loss: {val_loss}, Perplexity: {val_perplexity}')
        scheduler.step()

        sample_text = "[BAD] The chicken was dry, and the sauce was bland. Suggestion:"
        generated_sample = generate_text(model, tokenizer, sample_text)
        logging.info(f'Generated text after Epoch {epoch}: {generated_sample}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            logging.info("Validation loss improved, saving model...")
            model.save_pretrained('models/model_epoch_best')
            tokenizer.save_pretrained('models/model_epoch_best')
        else:
            patience_counter += 1
            logging.info(f"No improvement in validation loss. Patience counter: {patience_counter}")
            if patience_counter >= patience:
                logging.info("Early stopping due to no improvement in validation loss.")
                break

    model.save_pretrained('models/model_epoch_final')
    tokenizer.save_pretrained('models/model_epoch_final')
    logging.info("Model training completed and saved.")

if __name__ == "__main__":
    main()
