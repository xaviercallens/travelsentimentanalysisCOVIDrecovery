import tweepy
import csv
import re
from datetime import datetime, timezone

import twitter_credentials



class TwitterAuthenticator:
    """Authenticates to Twitter API."""

    def authenticate(self):
        auth = tweepy.OAuthHandler(
            twitter_credentials.CONSUMER_KEY,
            twitter_credentials.CONSUMER_SECRET
        )
        auth.set_access_token(
            twitter_credentials.ACCESS_TOKEN,
            twitter_credentials.ACCESS_TOKEN_SECRET
        )
        return auth


class StreamListener(tweepy.StreamListener):
    """Receives and handles Twitter's stream of data."""

    def __init__(self, rm):
        """@param rm (RunManager)"""

        super(StreamListener, self).__init__()

        self.c = rm.c
        self.rm = rm

        self.util = TweetsUtility()

    def on_status(self, status):
        """Save tweets in csv file with various information.
        Ignore retweets by directly getting the original tweet
        and ignore tweets older than one day.
        """

        status = self.util.get_original_status(status)

        # Ignore duplicates.
        if status.id in self.rm.collected_tweets_ids:
            return True

        # Ignore tweets older than one day.
        elapsed_time = self.util.get_elapsed_time(status.created_at)
        if elapsed_time > 24*3600:
            return True

        # tweet = self.util.clean_tweet(self.util.get_full_tweet(status))
        tweet = self.util.get_full_tweet(status)

        # Save tweet in csv file.
        with open(self.c.tweets_filename, 'a') as f:
            writer = csv.writer(f)
            unicode_encode_error = False
            try:
                writer.writerow([
                    tweet.encode('utf-8'),
                    status.id,
                    status.created_at,
                    self.util.get_country(status).encode('utf-8'),
                    status.user.screen_name.encode('utf-8'),
                    status.is_quote_status,
                    self.util.is_reply(status),
                    status.favorite_count,
                    status.retweet_count,
                    status.quote_count,
                    status.reply_count,
                    self.util.get_hashtags(status),
                    self.util.get_mentions(status)
                ])
                self.rm.collected_tweets_ids.add(status.id)
            except UnicodeEncodeError:
                unicode_encode_error = True
                self.rm.list_encode_error_tweets.append(tweet)

        self.rm.stream_tick(tweet, status, elapsed_time, unicode_encode_error)

        # Disconnect if number tweets limit reached.
        if self.c.nb_tweets_limit and self.rm.stream_tweets_count >= self.c.nb_tweets_limit:
            return False
    
    def on_error(self, status_code):
        print(status_code)
        # Disconnect the stream if being rate limited.
        if status_code == 420:
            # Returning False in on_data disconnects the stream.
            return False


class TweetsUtility:

    def get_original_status(self, status):
        """If status is a retweet, return the original status.
        Else, return the status.
        """

        if hasattr(status, "retweeted_status"):
            return status.retweeted_status
        else:
            return status
    
    def get_elapsed_time(self, created_at):
        elapsed_time = datetime.now(timezone.utc) - created_at.replace(tzinfo=timezone.utc)
        return elapsed_time.total_seconds()
    
    def get_full_tweet(self, status):
        """Return the full text of the tweet.
        Source: http://docs.tweepy.org/en/latest/extended_tweets.html
        """

        if hasattr(status, "retweeted_status"):  # Check if Retweet
            try:
                return status.retweeted_status.extended_tweet["full_text"]
            except AttributeError:
                return status.retweeted_status.text
        else:
            try:
                return status.extended_tweet["full_text"]
            except AttributeError:
                return status.text
    
    def clean_tweet(self, tweet):
        return ' '.join(re.sub("([^\w'’,.]+)|(\w+:\/\/\S+)", " ", tweet).split())
    
    def get_country(self, status):
        return status.place.country if status.place is not None else 'null'
    
    def get_hashtags(self, status):
        return [hashtag['text'].encode('utf-8') for hashtag in status.entities['hashtags']]

    def get_mentions(self, status):
        return [mention['screen_name'].encode('utf-8') for mention in status.entities['user_mentions']]
    
    def is_reply(self, status):
        return status.in_reply_to_status_id is not None


class TwitterStreamer:
    """Connects to Twitter's stream."""

    def __init__(self, rm):
        """@param rm (RunManager)"""

        self.c = rm.c
        self.rm = rm

        self.auth = TwitterAuthenticator().authenticate()
        self.listener = StreamListener(rm)
    
    def stream_tweets(self, is_async=True):
        stream = tweepy.Stream(self.auth, self.listener, is_async=is_async)
        self.rm.begin_stream()
        stream.filter(track=self.c.keywords, languages=self.c.languages)