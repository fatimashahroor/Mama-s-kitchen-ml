import openai
import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv('../openai.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

def count_tokens(text, model="gpt-3.5-turbo"):
    """
    Utility function to estimate the number of tokens for a given text.
    GPT-3.5-turbo uses around 4 tokens per word on average.
    """
    return len(text.split()) * 4
def classify_and_suggest_multiple_reviews(reviews, max_tokens=4000):
    """
    Dynamically process multiple reviews in batches based on the token limit.
    Only generate suggestions for bad reviews.
    """
    restricted_words = ['restaurants', 'restaurant', 'your', 'you']

    restriction_note = f"Do not use the following words in your suggestions: {', '.join(restricted_words)}.\n\n"
    
    token_limit = max_tokens - 500
    prompt_base = f"Classify each of the following reviews as 'good' or 'bad'. If it's bad, suggest how to improve the dish. {restriction_note}\n\n"
    current_prompt = prompt_base
    all_responses = []
    for i, review in enumerate(reviews):
        review_prompt = f"Review {i+1}: {review}\n"
        
        token_count = count_tokens(current_prompt + review_prompt)
        
        if token_count < token_limit:
            current_prompt += review_prompt
        else:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": current_prompt}],
                max_tokens=500,
                temperature=0.7,
                top_p=0.9
            )
            all_responses.append(response['choices'][0]['message']['content'].strip())
            
            current_prompt = prompt_base + review_prompt

    if current_prompt != prompt_base:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": current_prompt}],
            max_tokens=500,
            temperature=0.7,
            top_p=0.9
        )
        print(f"Received final batch response: {response}")  # Debugging line
        all_responses.append(response['choices'][0]['message']['content'].strip())

    return all_responses

