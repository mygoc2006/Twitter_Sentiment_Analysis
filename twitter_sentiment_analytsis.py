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

# remove all www links 
rem_www = r'www.\w+.\w+[^ ]+'
train['tweet'] = train['tweet'].str.replace(rem_www,'')

train['tweet'].head()

# Fixing shorthands 
train['tweet'] = train['tweet'].str.replace('u','you')
train['tweet'] = train['tweet'].str.replace('cus','becuase')

# function to clean data 
def tweet_cleaning(text):
  soup = bs(text,'lxml')
  soup = soup.get_text()
  words = tok.tokenize(soup)
  words = detok.detokenize(words, return_str = True)
  words =  (" ".join(words)).strip()
  words = re.sub(join_3,' ', soup)
  return words

notclean_tweet = train.tweet[:]
cleaned_tweet = []
for t in notclean_tweet:
  cleaned_tweet.append(tweet_cleaning(t))
cleaned_tweet

# saving cleaned data
clean_df = pd.DataFrame(cleaned_tweet, columns = ['tweet'])
clean_df['Result'] = train.label
clean_df

# part 2 - Word Cloud using the Python Library WordCloud
hate_tweets = clean_df[clean_df.Result == 1]
hate_string = []
for t in hate_tweets.tweet:
  hate_string.append(t)
hate_string = pd.Series(hate_string).str.cat(sep = ' ')

# word cloud for positive words
pos_tweets = clean_df[clean_df.Result == 0]
pos_strings = []
for t in pos_tweets.tweet:
    pos_strings.append(t)
pos_strings = pd.Series(pos_strings).str.cat(sep= ' ')

from wordcloud import WordCloud
wordcloud = WordCloud(width =1600, height =1000, max_font_size=200).generate(pos_strings)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation = "bilinear")
plt.show()

#word cloud for negative words 
neg_tweets = clean_df[clean_df.Result == 1]
neg_strings = []
for t in neg_tweets.tweet:
    neg_strings.append(t)
neg_strings = pd.Series(neg_strings).str.cat(sep= ' ')

from wordcloud import WordCloud
wordcloud = WordCloud(width =1600, height =1000, max_font_size=200).generate(neg_strings)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation = "bilinear")
plt.show()

# identify commonly used words like user as shown in the wordcloud 
freq = pd.Series(' '.join(clean_df['tweet']).split()).value_counts()[:10]
freq

# remove commonly used words
freq = list(freq.index)
clean_df['tweet'] = clean_df['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
clean_df['tweet'].head()


# Remove rarely occuring words
freq = pd.Series(' '.join(clean_df['tweet']).split()).value_counts()[-20:]
freq
freq = list(freq.index)
clean_df['tweet'] = clean_df['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
clean_df['tweet'].head()

# Spelling correction
# conda install -c conda-forge textblob
from textblob import TextBlob
clean_df['tweet'][:5].apply(lambda x: str(TextBlob(x).correct()))