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

