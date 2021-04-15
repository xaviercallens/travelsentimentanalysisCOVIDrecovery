from twitter_streamer import TwitterStreamer
from runManager import RunManager
from read_keywords import get_keywords



class Configuration:
    """Contains all parameters of the program."""

    def __init__(self):

        folder = 'data/'

        # CSV file name of collected tweets
        self.tweets_filename = folder + 'stream_tweets.csv'

        # To create a new file or append to an existing file.
        self.new_file = True

        # Number of tweets to collect. Set to None if infinite.
        self.nb_tweets_limit = None

        # Parameters for stream filter.
        keywords_filename = folder + 'keywords_1.txt'
        self.keywords = get_keywords(keywords_filename)
        self.languages = None

        # Number of tweets collected between each print of a stream's progression.
        self.stream_prints_interval = 1



if __name__ == '__main__':

    c = Configuration()
    rm = RunManager(c)

    streamer = TwitterStreamer(rm)
    streamer.stream_tweets(rm)