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
import textblob

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import nltk
from nltk.corpus import stopwords

from read_file import dataframe_from_file



class Textblob_analysis:

    def __init__(self, filename):
        self.d = dataframe_from_file(filename)
        self.compute_sentiment_scores()

    def clean_text(self, text):
        return ' '.join(re.sub("([^\w'’,.]+)|(\w+:\/\/\S+)", " ", text).split())

    def sentiment_score(self, text):
        analysis = textblob.TextBlob(self.clean_text(text))
        return analysis.sentiment.polarity
    
    def discrete_sentiment(self, sentiment_score):
        if sentiment_score > 0: return 1
        elif sentiment_score == 0: return 0
        else: return -1
    
    def compute_sentiment_scores(self):
        self.d['sentiment'] = [self.sentiment_score(text) for text in self.d['text']]
        self.d['discrete_sentiment'] = [self.discrete_sentiment(sentiment) for sentiment in self.d['sentiment']]
    
    def sentiment_distr(self):
        """Plot distribution of sentiment."""
        plt.figure(figsize=(12,6))
        p1 = sns.kdeplot(self.d['sentiment'], shade=True, color="g")\
            .set_title('Distribution of Sentiment')

    def discrete_sentiment_distr(self):
        ax = self.d['discrete_sentiment'].plot.hist(bins=3)

    def print_texts_with_one_sentiment(self, sentiment_score, nb_texts):
        pd.options.display.max_colwidth = 1000
        texts = self.d.loc[self.d['discrete_sentiment'] == sentiment_score]['text']

        if sentiment_score == 1: sentiment = 'POSITIVE'
        elif sentiment_score == 0: sentiment = 'NEUTRAL'
        elif sentiment_score == -1: sentiment = 'NEGATIVE'
        else: sentiment = 'ERROR'

        print('_'*50)
        print('\n' + '*' * 15 + f" {sentiment} TWEETS " + '*' * 15)
        for text in texts[:nb_texts]:
            print('_'*50)
            print()
            print(text)



if __name__ == '__main__':

    filename = 'data/stream_tweets_3k.csv'

    ta = Textblob_analysis(filename)

    ta.discrete_sentiment_distr()
    ta.sentiment_distr()

    ta.print_texts_with_one_sentiment(sentiment_score=1, nb_texts=3)
    ta.print_texts_with_one_sentiment(sentiment_score=-1, nb_texts=3)
    ta.print_texts_with_one_sentiment(sentiment_score=0, nb_texts=3)