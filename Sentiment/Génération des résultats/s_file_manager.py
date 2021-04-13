import numpy as np

def save_array(array, filename):
    with open(filename, 'wb') as f:
        np.save(f, array)

def load_array(filename):
    with open(filename, 'rb') as f:
        array = np.load(f)
    return array

def get_sentiment_filename(folder_name, tweets_filename, model_name):
    return folder_name + f'{tweets_filename} - sentiments - {model_name}'

def save_sentiments(sentiments, folder_name, tweets_filename, model_name):
    """Save sentiments array in the file named with get_sentiment_filename function."""
    sentiments = np.array(sentiments, dtype=np.int8)
    save_array(sentiments, get_sentiment_filename(folder_name, tweets_filename, model_name))

def load_sentiments(folder_name, tweets_filename, model_name):
    """Load the file containing sentiment array named with get_sentiment_filename function."""
    return load_array(get_sentiment_filename(folder_name, tweets_filename, model_name))

def get_clusters_filename(folder_name, tweets_filename):
    return folder_name + f'{tweets_filename} - clusters'

def load_clusters(folder_name, tweets_filename):
    return load_array(get_clusters_filename(folder_name, tweets_filename))