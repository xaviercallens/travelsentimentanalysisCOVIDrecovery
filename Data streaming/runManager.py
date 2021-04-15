import time
from datetime import timedelta
import csv
import pandas as pd


class RunManager:
    """Manages runs of the program and tracks runs' progressions."""

    def __init__(self, c):
        """
        @param c (Configuration)
        """

        self.c = c

        self.stream_tweets_count = 0
        self.stream_start_time = 0
        self.stream_duration = 0

        if c.new_file:
            self.collected_tweets_ids = set()
        else:
            with open(self.c.tweets_filename, 'r') as f:
                self.collected_tweets_ids = set(pd.read_csv(f, skipinitialspace=True, usecols=['id']).iloc[:, 0])
            self.stream_tweets_count = len(self.collected_tweets_ids)

        # List of tweets not saved because of a unicode encode error.
        self.list_encode_error_tweets = []
    
    def begin_stream(self):
        # For a new csv file, we write the columns names to the file.
        if self.c.new_file:
            with open(self.c.tweets_filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'text', 'id', 'date', 'country', 'user_screen_name',
                    'is_quote', 'is_reply',
                    'favorite_count', 'retweet_count', 'quote_count', 'reply_count',
                    'hashtags', 'mentions'
                ])

        self.stream_start_time = time.time()
    
    def stream_tick(self, tweet, status, elapsed_time, unicode_encode_error):

        if not unicode_encode_error:
            self.stream_tweets_count += 1
        self.stream_duration = time.time() - self.stream_start_time

        # Print stream progression.
        if self.stream_tweets_count % self.c.stream_prints_interval == 0 \
        or self.stream_tweets_count == self.c.nb_tweets_limit:
            self.print_stream_progression(tweet, status, elapsed_time, unicode_encode_error)
        
        # At the end, print tweets in list_encode_error_tweets.
        if self.stream_tweets_count == self.c.nb_tweets_limit:
            self.print_encode_error_tweets(tweet)
    
    def print_stream_progression(self, tweet, status, elapsed_time, unicode_encode_error):

        str_stream_duration = str(timedelta(seconds=round(self.stream_duration)))
        str_unicode_encore_error = "\n\nUNICODE ENCODE ERROR " if unicode_encode_error else ''
        str_elapsed_time = str(timedelta(seconds=round(elapsed_time)))

        print(
            ('_' * 50) +
            str_unicode_encore_error +

            f"\n\ncollected tweets: {self.stream_tweets_count} / {self.c.nb_tweets_limit}"
            f" | time: {str_stream_duration}"

            f"\n\n{tweet}"
            + '\n\n' + ('*' * 20) +
            f"\n\nfav: {status.favorite_count} | rt: {status.retweet_count}"
            f" | q: {status.quote_count} | r: {status.reply_count}"
            f"\n\npast time: {str_elapsed_time}"
            f"\n\nuser: {status.user.screen_name}"
        )
    
    def print_encode_error_tweets(self, tweet):
        for tweet in self.list_encode_error_tweets:
            print(
                '\n' + '*' * 50 +
                "\n\nNOT SAVED BECAUSE OF UNICODE ENCODE ERROR" +
                f" ({len(self.list_encode_error_tweets)} tweets)"
                f"\n\n{tweet}"
            )