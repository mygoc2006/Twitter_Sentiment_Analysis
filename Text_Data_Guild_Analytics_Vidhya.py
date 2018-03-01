# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 13:27:43 2018

@author: etagbett
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as bs
import re

#Load Data into a panda Dataframe
train = pd.read_csv('train_tweets.csv')
test = pd.read_csv('test_tweets.csv')

#Examine the data
train.head(5)

#1.Determining Number of Characters
train['word_count'] = train['tweet'].apply(lambda x: len(str(x).split(" ")))
train[['tweet', 'word_count']].min()

#1.2 Average word length
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words) /len(words))

train['avg_word'] = train['tweet'].apply(lambda x: avg_word(x))
train[['tweet','avg_word']].head()

#1.3 Removing StopWords
# a stop word is a commonly used word (such as "the")
from nltk.corpus import stopwords
stop = stopwords.words('english')

train['stopwords'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x in stop]))
train[['tweet','stopwords']].head()

#Number of special Characters
train['hastags'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
train[['tweet','hastags']].head()

# Number of Numerics
train['numerics'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
train[['tweet','numerics']].head()

# Number of Uppercase words
train['upper'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
train[['tweet','upper']].head()


# BASIC PRE-PROCESSING
# converting all to lowercase
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
train['tweet'].head()

# Removing Punctuations
#[^] - matches any single character not in square bracket
train['tweet'] = train['tweet'].str.replace('[^\w\s]','')
train['tweet'].head()

# Removal of Stop Words
from nltk.corpus import stopwords
stop = stopwords.words('english')
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
train['tweet'].head()


# identify commonly used words
freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[:10]
freq

# remove commonly used words
freq = list(freq.index)
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
train['tweet'].head()

# Remove rarely occuring words
freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[-10:]
freq
freq = list(freq.index)
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
train['tweet'].head()


# Spelling correction
# conda install -c conda-forge textblob
from textblob import TextBlob
train['tweet'][:5].apply(lambda x: str(TextBlob(x).correct()))


# Tokenization
#Tokenization refers to dividing the text into a sequence of words or sentences
TextBlob(train['tweet'][1]).words

#2.8 Stemming
#Stemming refers to the removal of suffices, like “ing”, “ly”, “s”, etc. by a simple rule-based approach
from nltk.stem import PorterStemmer
st = PorterStemmer()
train['tweet'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

#Lemmatization
#Lemmatization is a more effective option than stemming because it converts the word into its root word, rather than just stripping the suffices. It makes use of the vocabulary and does a morphological analysis to obtain the root word. Therefore, we usually prefer using lemmatization over stemming

from textblob import Word
train['tweet'] = train['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
train['tweet'].head(10)


#Advance Text Processning
# N-grams are the combination of multiple words used together. Ngrams with N=1 are called unigrams. Similarly, bigrams (N=2), trigrams (N=3) and so on can also be used. Unigrams do not usually contain as much information as compared to bigrams and trigrams. The basic principle behind n-grams is that they capture the language structure, like what letter or word is likely to follow the given one. The longer the n-gram (the higher the n), the more context you have to work with. Optimum length really depends on the application – if your n-grams are too short, you may fail to capture important differences. On the other hand, if they are too long, you may fail to capture the “general knowledge” and only stick to particular cases.

# N-grams
TextBlob(train['tweet'][0]).ngrams(2)

#Term Frequency
# Term frequency is simply the ratio of the count of a word present in a sentence, to the length of the sentence.
# Therefore, we can generalize term frequency as
# TF = (Number of times term T appears in the particular row) / (number of terms in that row)

tf1 = (train['tweet'][5:6]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
tf1