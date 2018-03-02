# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:58:38 2018

@author: etagbett
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as bs
import re

train = pd.read_csv('train_tweets.csv')
test = pd.read_csv('test_tweets.csv')

# Explore the data to understand the attributes and features associated with the data
# there are 3 features ID, Lable and tweet. tweet is an object that contains alphanumerica and special characters
train.head(20)

# there are no missing values in the train data
# test data also has no missing values
train.info()

# lets examine the length of the string in the tweet column
train['tweet_len'] = [len(t) for t in train.tweet]
train['string_len'] = train['tweet'].apply(lambda t: len(str(t).split(" ")))
train.tweet_len
train.string_len

# lets plot the pre_clean data to examine the distribution
plt.subplots(figsize =(5,4))
plt.boxplot(train.tweet_len, '0','gD')
plt.show()

#1 Average word length
def avg_word(sentence):
  tweet_split = sentence.split()
  return (sum(len(t) for t in tweet_split) / len(tweet_split))

train['avg_word'] = train['tweet'].apply(lambda t: avg_word(t))
train[['tweet','avg_word']].head()

# HTML Encoding
#string = train.tweet[1]
#text_decode = bs(string, 'lxml').get_text()

# Removing URL Links
# There are no URL Links in the tweet text
# re.sub('https?://[A-Za-z0-9./]+', '', train.tweet[13])

# Removing special characters
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize.moses import MosesDetokenizer
tok = WordPunctTokenizer()
detok = MosesDetokenizer()

# removes all strings that starts with @
rem_1 = r"@[\w]+"

# removes all special characters except '
rem_3 = '[^\w\']+'
join_3 = r'|'.join((rem_1, rem_3))

# re.findall(r'www. *\w+. ?com',  'so glad my #workout includes smoke breaks... www. smokeweedeatbacon. com #weed
# re.findall(r'www *\w+ ?com',  'www sexgirl com hk   hardcore eye opener')
# re.findall(r'www\w+com',  'porn vids wwwsmallgirlsexcom')
# re.findall(r'www.\w+.com',  'www.flybcc.com')
# re.findall(r'www.\w+',  'couple having sex www.drunk singapore girl get fuck ')
# re.findall(r'www.\w+/\w+',  '@user just run 10kms for pour donner: www.alvarum/heloiseetlespremas')

# remove all www links 
rem_www = r'www.\w+.\w+[^ ]+'
train['tweet'] = train['tweet'].str.replace(rem_www,'')

train['tweet'][23844]


def tweet_cleaning(text):
  soup = bs(text,'lxml')
  soup = soup.get_text()
  words = tok.tokenize(soup)
  words = detok.detokenize(words, return_str = True)
  words =  (" ".join(words)).strip()
  words = re.sub(join_3,' ', soup)
#  words = words.replace(" '", "'")
#  words = words.replace("' ", "'")
#  words = words.replace("&", "and")
#  words = words.replace(" !", "!")
#  words = words.replace(" .", "")
#  words = words.replace(". ", ".")
#  words = re.sub(r'www. *\w+. ?com','', words)
#  words = re.sub(r'www *\w+ ?com','', words)
#  words = re.sub(r'www\w+com','', words)
#  words = re.sub(r'www.\w+.com','', words)
# re.sub(" ?' ?", "'", words)
#  output = "'".join(output.split(" '"))
#  output = "'".join(output.split("' "))
  return words

notclean_tweet = train.tweet[11110:11119]
cleaned_tweet = []
for t in notclean_tweet:
  cleaned_tweet.append(tweet_cleaning(t))
cleaned_tweet

# saving cleaned data
clean_df = pd.DataFrame(cleaned_tweet, columns = ['tweet'])
clean_df['Result'] = train.label
clean_df
clean_df.info()
# text1 =  train.tweet[1619]
# pattern = "[^\w ]+ "
# soup =  bs(text1,'lxml')
# soup = soup.get_text()
# text1 = re.sub(pattern,'', soup)
# tk = tok.tokenize(text1)
# #output =  (' '.join(tk)).strip()
# output = detok.detokenize(tk, return_str = True )
# re.sub(" ?' ?", "'", output)

# part 2 - Word Cloud using the Python Library WordCloud
hate_tweets = clean_df[clean_df.Result == 1]
hate_string = []
for t in hate_tweets.tweet:
  hate_string.append(t)
hate_string = pd.Series(hate_string).str.cat(sep = ' ')

from wordcloud import WordCloud


