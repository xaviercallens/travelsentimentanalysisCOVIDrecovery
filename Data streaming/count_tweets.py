import pandas as pd
from configuration import Configuration


def count_tweets(tweets_filename):
    with open(tweets_filename, 'r') as f:
        nb_tweets = len(pd.read_csv(f, skipinitialspace=True, usecols=['id']).iloc[:, 0])
    return nb_tweets


if __name__ == '__main__':

    c = Configuration()
    tweets_filename = c.tweets_filename

    nb_tweets = count_tweets(tweets_filename)
    print(f"nb tweets: {nb_tweets}")