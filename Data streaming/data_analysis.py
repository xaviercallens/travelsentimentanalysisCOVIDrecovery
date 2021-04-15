import re
import string
import numpy as np 
import random
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from collections import Counter
from datetime import datetime

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import nltk
from nltk.corpus import stopwords

from read_file import dataframe_from_file


class DataAnalysis:
    '''Various tools to analyze data.'''

    def __init__(self, filename):
        self.d = dataframe_from_file(filename)
        self.d['text'] = self.d['text'].apply(self.clean_text)
    
    def clean_text(self, text):
        '''Make text lowercase, remove text in square brackets, remove links, remove punctuation
        and remove words containing numbers.'''

        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text

    def print_len(self):
        '''Print number of unique texts and total number of texts.'''
        print(
            f"\nunique counts: {len(self.d['text'].value_counts())}"
            f"\ntotal counts: {len(self.d)}"
        )

    def nb_char_distr(self):
        """Plot distribution of number of characters in d.text"""
        num_char = self.d['text'].apply(len)
        plt.figure(figsize=(12,6))
        p1 = sns.kdeplot(num_char, shade=True, color="r")\
            .set_title('Distribution of Number of Characters')
    
    def nb_words_distr(self):
        """Plot distribution of number of words in d.text"""
        num_words = self.d['text'].apply(lambda t: len(t.split()))
        plt.figure(figsize=(12,6))
        p1 = sns.kdeplot(num_words, shade=True, color="g")\
            .set_title('Distribution of Number of Words')

    def nb_tweets_over_time(self, frequence='1M'):
        """Print the histogram of the number of tweets over time.
        @param frequence (str): width of the bars of the histogram
        'A, Y': year end
        'M': month end
        'W': weekly
        'D': calendar day
        'H': hourly
        'T, min': minutely
        'S': secondly
        """

        dates = pd.DataFrame(self.d['date'].apply(pd.Timestamp))
        dates.set_index('date', drop=False, inplace=True)
        dates.groupby(pd.Grouper(freq=frequence)).count().plot(
            figsize=(12,6), kind='bar', color='tab:blue',
            title='Number of Tweets over Time'
        )

    def remove_stopword(self, text):
        return [word for word in text if word not in stopwords.words('english')]

    def most_common_words(self, nb_words=20, remove_stopwords=True):
        '''Display most common words in d.text'''
        
        words_lists = self.d['text'].apply(lambda t: t.split())
        if remove_stopwords:
            words_lists = words_lists.apply(self.remove_stopword)

        top = Counter([word for words in words_lists for word in words])
        top_words = pd.DataFrame(top.most_common(nb_words))
        top_words.columns = ['common_words', 'count']
        top_words.style.background_gradient(cmap='Blues')
        if remove_stopwords: title = 'Commmon Words (no stopwords)'
        else: title = 'Commmon Words'
        fig = px.bar(top_words, x="count", y="common_words", title=title, orientation='h', 
             width=700, height=700, color='common_words')
        fig.show()


if __name__ == '__main__':

    filename = 'data/stream_tweets.csv'

    da = DataAnalysis(filename)

    da.print_len()
    print()
    da.nb_char_distr()
    da.nb_words_distr()
    da.nb_tweets_over_time(frequence='1H')
    da.most_common_words(nb_words=20, remove_stopwords=False)
    da.most_common_words(nb_words=20, remove_stopwords=True)