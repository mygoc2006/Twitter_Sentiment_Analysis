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

#Explore the data to understand the attributes and features associated with the data
#there are 3 features ID, Lable and tweet. tweet is an object that contains alphanumerica and special characters
train.head(20)

#there are no missing values in the train data
# test data also has no missing values
train.info()
test.info()

#lets examine the length of the string in the tweet column
train['tweet_len'] = [len(t) for t in train.tweet]

# (Min, Max) characters is (11, 274) and mean is 88
train.tweet_len
train.tweet_len.describe()

#Data Preperation
from pprint import pprint
tweet_dict = {
    'label':{
        'type':train.label.dtypes,
        'description':'sentiment class - 0: racist_speech, 1:postive_speech'},

    'tweet':{
        'type': train.tweet.dtype,
        'description': 'tweet text'
        },
    'tweet_len':{
        'type': train.tweet_len.dtype,
        'description': 'tweet length before cleaning'
        },
        'dataset_shape': train.shape
    }
pprint(tweet_dict)

#lets plot the pre_clean data to examine the distribution
# boxplot indicates outliers in your text length which aren't since twitter max text is now 280
plt.subplots(figsize =(5,4))
plt.boxplot(train.tweet_len, '0','gD')
plt.show()

#HTML Encoding
#we need DEcode html to general text to handle wierd text like #got7, angryð
string = train.tweet[1]
text_decode = bs(string, 'lxml')
get_text = text_decode.get_text()

# Removing @mention
train.tweet[609]
re.sub(r'www.[^ ]+', '', train.tweet[609])

#Removing URL Links
# There are no URL Links in the tweet text
#re.sub('https?://[A-Za-z0-9./]+', '', train.tweet[13])

#Removing non characters
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize.moses import MosesDetokenizer
tok = WordPunctTokenizer()
detok = MosesDetokenizer()

rem_1 = r"@[A-Za-z0-9]+"
rem_2 = '[^A-Za-z0-9\'\:\.\/\,\?\"\!\;\_\-\&\$]+'
rem_2 = '[^\w\'\:\.\/\,\?\"\!\;\_\-\&\$]+'
pattern = "[^\w ]+ "
regex_www = r'www.\w+.\w+[^ ]+'
join_3 = r'|'.join((rem_1, rem_2))

#re.findall(r'www. *\w+. ?com',  'so glad my #workout includes smoke breaks... www. smokeweedeatbacon. com #weed #bacon #fitness #sex #health #marijuana   #strength #living')
#re.findall(r'www *\w+ ?com',  'www sexgirl com hk   hardcore eye opener')
#re.findall(r'www\w+com',  'porn vids wwwsmallgirlsexcom')
#re.findall(r'www.\w+.com',  'www.flybcc.com')
#re.findall(r'www.\w+',  'couple having sex www.drunk singapore girl get fuck ')
#re.findall(r'www.\w+/\w+',  '@user just run 10kms for @user @user   #loveisall  pour donner: www.alvarum/heloiseetlespremas')




def tweet_cleaning(text):
  soup = bs(text,'lxml')
  soup = soup.get_text()
  words = tok.tokenize(soup)
  words = detok.detokenize(words, return_str = True)
  #words =  (" ".join(words)).strip()
  words = re.sub(join_3,' ', soup)
  words = words.replace(" '", "'")
  words = words.replace("' ", "'")
  words = words.replace("&", "and")
  words = words.replace(" !", "!")
#  words = words.replace(" .", "")
#  words = words.replace(". ", ".")
#  words = re.sub(r'www. *\w+. ?com','', words)
#  words = re.sub(r'www *\w+ ?com','', words)
#  words = re.sub(r'www\w+com','', words)
#  words = re.sub(r'www.\w+.com','', words)
  #re.sub(" ?' ?", "'", words)
#  output = "'".join(output.split(" '"))
#  output = "'".join(output.split("' "))
  return words

notclean_tweet = train.tweet[:]
cleaned_tweet = []
for t in notclean_tweet:
  cleaned_tweet.append(tweet_cleaning(t))
cleaned_tweet

#saving cleaned data
clean_df = pd.DataFrame(cleaned_tweet, columns = ['tweet'])
clean_df['Result'] = train.label
clean_df
clean_df.info()
#text1 =  train.tweet[1619]
#pattern = "[^\w ]+ "
#soup =  bs(text1,'lxml')
#soup = soup.get_text()
#text1 = re.sub(pattern,'', soup)
#tk = tok.tokenize(text1)
##output =  (' '.join(tk)).strip()
#output = detok.detokenize(tk, return_str = True )
#re.sub(" ?' ?", "'", output)

#part 2 - Word Cloud using the Python Library WordCloud
hate_tweets = clean_df[clean_df.Result == 1]
hate_string = []
for t in hate_tweets.tweet:
  hate_string.append(t)
hate_string = pd.Series(hate_string).str.cat(sep = ' ')

from wordcloud import WordCloud

#import featuretools
