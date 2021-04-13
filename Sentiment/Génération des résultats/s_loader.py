from read_file import dataframe_from_file
from s_file_manager import load_sentiments, load_clusters
import sa_file_manager

class Sentiment_loader:
    '''Load tweets and sentiments to be used for sentiment analysis.

    @param tweets_folder (str): folder containing the tweets files (add a '/' at the end)
    @param sentiments_folder (str): folder containing the sentiments files (add a '/' at the end)
    @param sentiment_analysis_folder (str): folder in which the sentiments analysis results will be saved (add a '/' at the end)
    @param tweets_filename (str): name of the file containing the tweets to analyze (without the extension .csv)
    @param model_name (str): name of the model which gave the sentiments scores to analyze
    @param clusters_folder (str): folder containing the clusters files (add a '/' at the end)
    
    Attributes:

    d (dataframe): collected tweets plus a column for sentiments and a column for each cluster
    (-1 for negative, 0 for neutral, 1 for positive) and another for weights

    clusters (np.array): 2D array where data at (i,j) is a boolean indicating whether
    tweet number j (tweets must be in the same order as in the tweets file) belongs to cluster i.'''

    sentiment_int2str = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}

    def __init__(self, tweets_folder, sentiments_folder, sentiment_analysis_folder, tweets_filename, model_name, clusters_folder=None):
        
        self.sentiment_analysis_folder = sentiment_analysis_folder
        self.tweets_filename = tweets_filename
        self.model_name = model_name

        # Initialize the dataframe for sentiment analysis.
        self.d = dataframe_from_file(tweets_folder + tweets_filename + '.csv')
        self.d['sentiment'] = load_sentiments(sentiments_folder, tweets_filename, model_name)
        # Weight of each tweet representing its importance. (+1 because we must count the tweet itself)
        self.d['weight'] = 1 + self.d['retweet_count']

        # Load clusters
        self.clusters = None
        if clusters_folder is not None:
            self.clusters = load_clusters(clusters_folder, tweets_filename)
    
    def save_sa(self, sa, sa_name):
        '''Save a sentiment analysis result (a pandas series or dataframe).'''
        sa_file_manager.save_sa(sa, self.sentiment_analysis_folder, self.tweets_filename, self.model_name, sa_name)