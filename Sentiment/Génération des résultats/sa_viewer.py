import matplotlib.pyplot as plt
import numpy as np

from sa_file_manager import get_sa_param_filename
from s_loader import Sentiment_loader
from sa_loader import Sentiment_analysis_loader



class Sentiment_analysis_viewer:
    '''Class regrouping all objects to plot sentiment analysis results.'''

    def __init__(self, *args, **kwargs):
        '''Parameters are the same as of Sentiment_loader class.'''
        ld = Sentiment_analysis_loader(*args, **kwargs)
        self.sentiment_distribution = Sentiment_distribution(ld)
        self.sentiment_over_time = Sentiment_over_time(ld)
        self.text_by_sentiment = Text_by_sentiment(ld)


class Sentiment_analysis:
    '''Parent class for plotting sentiment analysis results.'''

    def __init__(self, sa_loader):
        self.ld = sa_loader
    
    def plot(self, plot_one_model, figsize, figrow):
        '''Plot graphs given by plot_one_model for all models.
        @param figsize: the size of each subplot'''
        figcol = int(np.ceil(len(self.ld.models_names) / figrow))
        figsize = (figsize[0] * figcol, figsize[1] * figrow)
        for i, model_name in enumerate(self.ld.models_names):
            ax = plt.subplot(figrow, figcol, i+1)
            plot_one_model(model_name, ax, figsize)
        plt.show()


class Sentiment_distribution(Sentiment_analysis):
    '''Class for plotting sentiment distribution.
    The distribution is the number or the percentage of tweets for each sentiment.'''

    def __init__(self, ld):
        Sentiment_analysis.__init__(self, ld)
    
    def plot(self, show_percentage=True, figsize=(6,4), figrow=1):
        '''Plot sentiment distribution for all models.'''
        Sentiment_analysis.plot(
            self,
            lambda model_name, ax, figsize: self.plot_one_model(model_name, ax, figsize, show_percentage),
            figsize, figrow
        )
    
    def plot_one_model(self, model_name, ax, figsize, show_percentage):
        '''Plot sentiment distribution for one model.'''

        bars = self.ld.dict_sentiment_distribution[model_name].copy()
        if show_percentage: bars = self.get_percentage(bars)

        # Plot bars.
        bars.plot.bar(
            ax=ax,
            rot=0,
            color=['red', 'gray', 'green'],
            xlabel='',
            ylabel='Number of tweets' + (' (%)' if show_percentage else ''),
            title=f'Sentiment distribution - {model_name}',
            figsize=figsize
        )

        # Extend ymax so that the text diplayed later remains inside the graph.
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax*1.05)

        # Show count or percentage values over each bar.
        for p in ax.patches:
            ax.annotate(
                str(p.get_height()) + (' %' if show_percentage else ''),
                (p.get_x()+p.get_width()/2, p.get_height()),
                ha='center', xytext=(0, 4), textcoords='offset points'
            )
    
    def get_percentage(self, bars):
        '''Compute and return the percentages of the counts of the tweets for each sentiment.'''
        return (bars / bars.sum() * 100).round().astype('int32')


class Sentiment_over_time(Sentiment_analysis):
    '''Class for plotting sentiment over time.'''

    def __init__(self, ld):
        Sentiment_analysis.__init__(self, ld)
    
    def plot(self, param_idx, figsize=(8,5), figrow=1):
        '''Plot sentiment over time for all models.'''
        Sentiment_analysis.plot(
            self,
            lambda model_name, ax, figsize: self.plot_one_model(model_name, ax, figsize, param_idx),
            figsize, figrow
        )
    
    def plot_one_model(self, model_name, ax, figsize, param_idx):
        '''Plot sentiment over time for one model.'''

        sent_over_time = self.ld.lst_sentiment_over_time[param_idx][model_name].drop('volume', axis=1)

        # Plot sentiment.
        sent_over_time.plot(
            ax=ax,
            kind='line',
            xlabel='Date',
            ylabel='Sentiment',
            title=f'Sentiment over Time - {model_name}' + get_sa_param_filename(self.ld.sent_over_time_params[param_idx]),
            figsize=figsize
        )
        ax.legend()
        # Plot a horizontal line on y=0.
        ax.axhline(y=0, color='k', linestyle=':')
    
    def plot_volume(self, param_idx, figsize=(8,4)):
        '''Plot volume over time.'''

        vol_over_time = next(iter(self.ld.lst_sentiment_over_time[param_idx].values()))['volume']
        str_freq = self.ld.sent_over_time_params[param_idx]['freq']
        vol_over_time.plot(
            kind='line',
            xlabel='Date',
            ylabel='Volume',
            title=f'Volume over Time' + get_sa_param_filename({'freq': str_freq}),
            figsize=figsize
        )
        plt.show()


class Text_by_sentiment:
    '''Class for getting and printing tweet's texts by sentiment.'''

    def __init__(self, ld):
        self.ld = ld
    
    def get(self, model, sentiment, nb_texts=None):
        '''Return the series containing the wanted texts.
        @param sentiment (int): 1, 0, or -1
        @param nb_texts (int): set to None if want all'''

        texts = self.ld.dict_text_by_sentiment[model][Sentiment_loader.sentiment_int2str[sentiment]].copy()
        if nb_texts is None:
            return texts
        else:
            return texts[:min(nb_texts, len(texts))]
    
    def print(self, model, sentiment, nb_texts=None):
        '''Print the wanted texts.'''

        texts = self.get(model, sentiment, nb_texts)

        title = '#'*15 + f" {Sentiment_loader.sentiment_int2str[sentiment]} tweets - {model} " + '#'*15
        sep_line1 = '#'*len(title)
        sep_line2 = '_'*len(title)
        print('\n' + sep_line1 + '\n' + title + '\n' + sep_line1)
        for text in texts:
            print(sep_line2 + '\n\n' + text)
        print(sep_line2)



if __name__ == '__main__':

    vw = Sentiment_analysis_viewer(
        tweets_folder='data/',
        sentiment_analysis_folder='data/sentiment analysis/',
        tweets_filenames=['stream_tweets_Week4_181k'],
        models_names=['Naive Bayes - TEST'] * 1,
        sent_over_time_params=[
            {'freq': '1D', 'norm': 'volume'},
            {'freq': '1D', 'norm': 'None'}
        ]
    )

    # vw.sentiment_distribution.plot(show_percentage=True, figsize=(6,4), figrow=1)
    vw.sentiment_over_time.plot_volume(param_idx=1, figsize=(8,4))
    # vw.sentiment_over_time.plot(param_idx=1, figsize=(8,5), figrow=1)
    # vw.text_by_sentiment.print(model='Naive Bayes - TEST', sentiment=1, nb_texts=5)