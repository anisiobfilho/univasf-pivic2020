# PIVIC: "Um modelo computacional para identificação de notícias falsas sobre a Covid-19 no Brasil"
# Code: Crawler for Twitter
# Author: Anísio Pereira Batista Filho

import os
import tweepy as tw
import csv
from tqdm.auto import tqdm
import time
import timeit

start = timeit.default_timer()

# Twitter Developer Keys for crawler_pivic app
consumer_key = '<consumer_key>'
consumer_secret = '<consumer_secret>'
access_token = '<access_token>'
access_token_secret = '<access_token_secret>'

# Autgentication between Twitter Developer and this script
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# Create a file to write the tweets
csvFile = open('data/tweetsdatabase_pivic/tweetsdatabase_pivic.csv', 'a', encoding='utf-8')
#csvFile = open('data/tweetsdatabase_pivic/tweetsdatabase_testing.csv', 'a', encoding='utf-8')
csvWriter = csv.writer(csvFile)

# Define the search term
search_words = "covid OR covid19 OR coronavirus OR caronavac OR astrazeneca OR pfizer OR sputnik v OR sputnik OR sinovac OR johnson & johnson OR johnson&johnson OR jnj OR butantan OR fio cruz OR fiocruz OR oxford OR moderna OR butanvac OR ufpr OR universidade OR federal OR parana OR paraná OR versamune OR covshield"
new_search = search_words + " -filter:retweets"

# Collect tweets
tweets = tw.Cursor(api.search,
                       q=new_search,
                       lang='pt',
                       tweet_mode="extended",                    
                       result_type='mixed',
                       count=10,
                       since='2021-08-06',
                       until='2021-08-07'
                    ).items(100)

# Write a list of tweets in .csv
for tweet in tqdm(tweets, total=tweets.limit):
    #print (tweet.id, tweet.created_at, tweet.user.location, tweet.full_text)
    csvWriter.writerow([tweet.id, tweet.created_at, tweet.user.location, tweet.full_text])

end = timeit.default_timer()
print ('Duração: %f segundos' % (end - start))