import openai
import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv('../openai.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def count_tokens(text, model="gpt-3.5-turbo"):
    """
    Utility function to estimate the number of tokens for a given text.
    GPT-3.5-turbo uses around 4 tokens per word on average.
    """
    token_count = len(text.split()) * 4
    logging.debug(f"Token count for the given text is {token_count}")
    return token_count

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
            all_responses.append(response.choices[0].message)
            logging.debug(f"Response added for batch ending at review {i}: {response}")
            current_prompt = prompt_base + review_prompt

    if current_prompt != prompt_base:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": current_prompt}],
            max_tokens=500,
            temperature=0.7,
            top_p=0.9
        )
        all_responses.append(response.choices[0].message)
        logging.debug(f"Final batch response received: {response}")

    return all_responses

def extract_suggestions_from_responses(responses):
    """
    Extract and format suggestions from API responses.
    Handles variations in formats such as 'Bad -', 'Bad.', or 'Bad - suggest', and includes optional detailed introductory phrases.
    """
    suggestions = []
    for response in responses:
        content = response.get('content', '')
        
        # Adjust the regex to capture after 'Bad -', 'Bad.' or 'Bad - suggest'
        matches = re.finditer(r"Bad(?: - suggest)?[. - ]\s*(?:\n\nSuggestion:)?\s*(.+)", content, re.IGNORECASE)
        for match in matches:
            suggestion = match.group(1).strip()
            # Remove any residual leading prompts such as '- suggest' that may still be present
            suggestion = re.sub(r"^- To improve the dish,", "", suggestion, flags=re.IGNORECASE).strip()
            # Capitalize the first letter and ensure proper punctuation
            if suggestion:
                formatted_suggestion = suggestion[0].upper() + suggestion[1:]
                if not formatted_suggestion.endswith('.'):
                    formatted_suggestion += '.'
                suggestions.append(formatted_suggestion)

    # Log the extracted suggestions for debugging
    logging.debug(f"Extracted suggestions: {suggestions}")
    if suggestions:
        # Join all suggestions into a single paragraph
        paragraph = " ".join(suggestions)
        return paragraph
    else:
        return "No suggestions to improve the dish."



@app.route('/process_reviews', methods=['POST'])
def process_reviews():
    """
    API endpoint to process reviews. Expects a JSON payload with an array of reviews.
    Example request payload:
    {
        "reviews": ["The chicken was dry.", "The pasta was amazing."]
    }
    """
    try:
        data = request.json
        reviews = data.get('reviews', [])
        logging.debug(f"Received reviews: {reviews}")
        if not reviews:
            return jsonify({"error": "No reviews provided"}), 400
        responses = classify_and_suggest_multiple_reviews(reviews)
        suggestions_paragraph = extract_suggestions_from_responses(responses)
        return jsonify({"suggestions": suggestions_paragraph}), 200

    except Exception as e:
        logging.error(f"Error processing reviews: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='192.168.1.11', port=5000)
