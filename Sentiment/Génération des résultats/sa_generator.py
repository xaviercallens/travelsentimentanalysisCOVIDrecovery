import pandas as pd
import numpy as np
import datetime

from sa_file_manager import get_sa_param_filename



class Sentiment_analysis_generator:
    '''Class regrouping all objects to generate sentiment analysis results.'''

    def __init__(self, data_loader):
        '''@param data_loader (Sentiment_loader)'''
        self.sentiment_distribution = Sentiment_distribution(data_loader)
        self.sentiment_over_time = Sentiment_over_time(data_loader)
        self.text_by_sentiment = Text_by_sentiment(data_loader)


class Sentiment_distribution:
    '''Class to generate the distribution of the sentiment.'''

    def __init__(self, data_loader):
        self.ld = data_loader

    def save(self):
        '''Compute and save the results.'''
        self.ld.save_sa(self.get(), 'sentiment distribution')
    
    def get(self):
        '''Return the series with the sum of the weights of the tweets for each sentiment.'''
        bars = self.ld.d.groupby('sentiment')['weight'].sum().reindex([-1,0,1])
        bars.rename(self.ld.sentiment_int2str, inplace=True)
        return bars


class Sentiment_over_time:
    '''Class to generate graph of the evolution of the sentiment score over time.'''

    def __init__(self, data_loader):
        self.ld = data_loader

    def save(self, clusters_names=None, frequence='1D', norm_option=None, time_sleep_threshold=None):
        '''Compute and save the results.'''
        df = self.get(clusters_names, frequence, norm_option, time_sleep_threshold)
        self.ld.save_sa(df, 'sentiment over time' + get_sa_param_filename({'freq': frequence, 'norm': norm_option}))
    
    def get(self, clusters_names=None, frequence='1D', norm_option=None, time_sleep_threshold=None):
        '''Return the global sentiment score for each time window
        and the sentiment score of the clusters if they are given.

        @param clusters_names (list[str]): names of the clusters in the same order as in the clusters' array's rows.'''

        sent_over_time = pd.DataFrame()

        # Volume
        sent_over_time['volume'] = self.get_volume_over_time(frequence)

        # Global sentiment
        sent_over_time['global'] = self.get_series(self.ld.d['sentiment'], frequence, norm_option, time_sleep_threshold)

        # Clusters' sentiments
        if self.ld.clusters is not None:
            for i in range(len(self.ld.clusters)):
                sentiments = self.ld.d['sentiment'].copy()
                sentiments[np.logical_not(self.ld.clusters[i])] = 0
                sent_over_time[clusters_names[i]] = self.get_series(sentiments, frequence, norm_option, time_sleep_threshold)
        
        return sent_over_time
    
    def get_series(self, sentiments, frequence='1D', norm_option=None, time_sleep_threshold=None):
        '''Return the sentiment score for each time window.

        @param frequence (str): the length of the time windows in which tweets are grouped.
        'A, Y': year end
        'M': month end
        'W': weekly
        'D': calendar day
        'H': hourly
        'T, min': minutely
        'S': secondly
        Example: '1M' for one month

        @param norm_option (str or None): normalization option of the sentiment over time.
        Available options: - 'volume': divide the sum of each sentiment by the number of tweets in the time window.
                           - 'stream_time': divide the sum by the proportion of the time during which the stream ran
                             versus the frequence of the time windows.
                           - None: no normalization (just the sum).
        
        @param time_sleep_threshold (frequence format str): used for stream_time normalization.
        It is the duration between the dates of 2 consecutive tweets from which the stream
        is considered as having been paused.'''

        sent_by_date = pd.DataFrame(
            {'sentiment': sentiments.to_numpy(), 'weight': self.ld.d['weight'].to_numpy()},
            index=self.ld.d['date'].apply(pd.Timestamp)
        )
        # We remove neutral tweets.
        sent_by_date = sent_by_date[sent_by_date['sentiment'] != 0]
        sent_by_date['weighted_sentiment'] = sent_by_date['weight'] * sent_by_date['sentiment']
        sent_grouped = sent_by_date.groupby(pd.Grouper(freq=frequence))
        sent_over_time = sent_grouped['weighted_sentiment'].sum()

        if norm_option == 'volume':
            sent_over_time = sent_over_time / sent_grouped['weight'].sum()

        elif norm_option == 'stream_time':
            stream_durations_prop = sent_grouped.apply(
                lambda grp: self.get_stream_duration_prop(frequence, time_sleep_threshold, grp)
            )
            sent_over_time = sent_over_time / stream_durations_prop

        elif norm_option is not None:
            raise Exception(f"The option '{norm_option}' is not available for normalizing sentiment over time.")

        return sent_over_time
    
    def get_volume_over_time(self, frequence='1D'):
        '''Return the volume for each time window.'''
        volume_by_date = pd.Series(
            self.ld.d['weight'].to_numpy(),
            index=self.ld.d['date'].apply(pd.Timestamp)
        )
        return volume_by_date.groupby(pd.Grouper(freq=frequence)).sum()

    def get_stream_duration_prop(self, frequence, time_sleep_threshold, grp):
        '''Return the proportion of the time during which the stream was 
        running versus the frequence of the time windows.'''
        duration = self.get_stream_duration(time_sleep_threshold, grp)
        if duration <= datetime.timedelta(minutes=1): return 1
        else: return duration / pd.to_timedelta(frequence)

    def get_stream_duration(self, time_sleep_threshold, grp):
        '''Return the time during which the stream was running.'''
        duration = datetime.timedelta(0)
        # grp.index is the dates of the group sorted in ascending order.
        for i in range(len(grp.index)-1):
            dt = grp.index[i+1] - grp.index[i]
            if dt < pd.to_timedelta(time_sleep_threshold):
                duration += dt
        return duration


class Text_by_sentiment:
    '''Class to get texts of tweets by sentiment.'''

    def __init__(self, data_loader):
        self.ld = data_loader
    
    def save(self, sentiments=[1,0,-1], nb_texts=10):
        '''Compute and save the results.'''
        df = self.get(sentiments, nb_texts)
        self.ld.save_sa(df, 'text by sentiment')

    def get(self, sentiments=[1,0,-1], nb_texts=10):
        '''Return some texts of tweets with specified sentiments.
        @param sentiments (List[int]): sentiments of which we want to see some texts
        @param nb_texts (int): number of texts we want to see for each sentiment'''
        text_by_sentiment = pd.DataFrame()
        for sentiment in sentiments:
            text_by_sentiment[self.ld.sentiment_int2str[sentiment]] = \
                self.ld.d.loc[self.ld.d['sentiment'] == sentiment]['text'][:nb_texts].to_list()
        return text_by_sentiment



if __name__ == '__main__':

    from s_loader import Sentiment_loader
    loader = Sentiment_loader(
        tweets_folder='data/',
        sentiments_folder='data/sentiments/',
        sentiment_analysis_folder='data/sentiment analysis/',
        tweets_filename='stream_tweets_Week4_181k',
        model_name='Naive Bayes - TEST',
        clusters_folder='data/clusters/',
    )
    gen = Sentiment_analysis_generator(loader)

    clusters_names = list(range(len(loader.clusters)))

    gen.sentiment_distribution.save()
    gen.sentiment_over_time.save(clusters_names=clusters_names, frequence='1D', norm_option=None, time_sleep_threshold=None)
    gen.sentiment_over_time.save(clusters_names=clusters_names, frequence='1D', norm_option='volume', time_sleep_threshold=None)
    gen.text_by_sentiment.save(sentiments=[1,0,-1], nb_texts=10)