import pandas as pd
from datetime import datetime
from sa_file_manager import load_sa_from_models_tweets, get_sa_param_filename

class Sentiment_analysis_loader:
    '''Load dictionaries of sentiment anlaysis results to be used by the Sentiment_analysis_viewer class.

    @param tweets_folder (str): folder containing the tweets files (add a '/' at the end)
    @param sentiment_analysis_folder (str): folder containing the sentiment analysis results files (add a '/' at the end)
    @param tweets_filenames (list[str]): list of the names of the files containing the tweets to analyze (without the extension .csv)
    @param models_names (list[str]): list of the names of the models which gave the sentiments scores to analyze
    @param sent_over_time_params (list[dict[str:str]]): list of dict mapping parameter's names to their corresponding values (only string)

    Attributes:
        - dict_sentiment_distribution (dict[str: pandas series])
        - lst_sentiment_over_time (lst[dict[str: pandas series]]): each dict corresponds to one configuration in the list sent_over_time_params
        - dict_text_by_sentiment (dict[str: pandas dataframe])'''

    def __init__(self, tweets_folder, sentiment_analysis_folder, tweets_filenames, models_names, sent_over_time_params):
        
        self.models_names = models_names
        self.sent_over_time_params = sent_over_time_params

        load_sa = lambda sa_name, concatenate: load_sa_from_models_tweets(
            sentiment_analysis_folder, models_names, tweets_filenames, sa_name, concatenate
        )
        # To concatenate sentiment distributions, we take the sum so the distributions must be counts and not percentages.
        self.dict_sentiment_distribution = load_sa('sentiment distribution', sum)
        self.lst_sentiment_over_time = [load_sa('sentiment over time' + get_sa_param_filename(param), pd.concat) for param in sent_over_time_params]
        self.dict_text_by_sentiment = load_sa('text by sentiment', pd.concat)

        # Convert string dates of sentiment_over_time dataframe's indexes to datetime.
        for dict_df in self.lst_sentiment_over_time:
            for df in dict_df.values():
                df.index = pd.to_datetime(df.index)