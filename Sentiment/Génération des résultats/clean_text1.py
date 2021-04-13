import re

def clean_text(text):
    return ' '.join(re.sub("([^A-Za-z0-9'’,.]+)|(\w+:\/\/\S+)", " ", text).split())