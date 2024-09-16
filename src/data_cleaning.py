import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

def get_wordnet_pos(treebank_tag):
    """
    Convert Treebank POS tags to WordNet POS tags for better lemmatization.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
def normalize_text(text):
    """ 
    Normalize text by tokenizing, lemmatizing with part-of-speech tags,
    and selectively retaining significant parts of speech.
    """
    if pd.isna(text):
        return ""  
    important_stop_words = {'not', 'more', 'very', 'no', 'without', 'less', "and", "n't", "won't", "wouldn't", "isn't", "wasn't", "didn't", "to", "of", "but", "yet", "although", "though"}
    stop_words_set = set(stopwords.words('english')) - important_stop_words

    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text.lower())
    tagged_words = pos_tag(words)

    lemmatized_words = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag)) 
        for word, tag in tagged_words 
        if word not in stop_words_set or word in important_stop_words
    ]
    cleaned_words = [word for word, tag in pos_tag(lemmatized_words) if tag.startswith(('NN', 'VB', 'JJ', 'RB')) or word in important_stop_words]

    return ' '.join(cleaned_words)
def load_and_clean_data(filepath):
    """ 
    Load data from CSV and clean only the 'Improvement_Suggestion' field. 
    """
    df = pd.read_csv(filepath)    
    df['Cleaned_Suggestion'] = df['Improvement_Suggestion'].apply(normalize_text)
    
    return df
def save_cleaned_data(df, output_filepath):
    """ 
    Save the cleaned data to a new CSV file. 
    """
    df.to_csv(output_filepath, index=False)
