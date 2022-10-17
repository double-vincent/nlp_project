from bs4 import BeautifulSoup as soupify
from os import path
from requests import get
from nltk.tokenize.toktok import ToktokTokenizer
from nltk import PorterStemmer
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

import pandas as pd
import numpy as np

import re
import unicodedata


def basic_clean(text):
    """ 
    Purpose:
        
    ---
    Parameters:
        
    ---
    Returns:

    """
    text = text.lower()

    text = unicodedata.normalize('NFKD', text)\
                        .encode('ascii', 'ignore')\
                        .decode('utf-8', 'ignore')
    
    text = re.sub(r"[^a-z0-9'\s]", '', text)
    return text


def tokenize(text):
    """ 
    Purpose:
        takes a string and tokenizes all words in t
    ---
    Parameters:
        
    ---
    Returns:

    """ 
    tokenizer = ToktokTokenizer()

    text = tokenizer.tokenize(text, return_str=True)

    return text    

def stem(text):
    """ 
    Purpose:
        to apply stemming to input text
    ---
    Parameters:
        text: the text to be stemmed
    ---
    Returns:
        text: text that has had stemming applied to it
    """

    #create the nltk stemmer object
    ps = PorterStemmer()    

    stems = [ps.stem(word) for word in text.split()]
    text = ' '.join(stems)

    return text

def lemmatize(text):
    """ 
    Purpose:
        applies lemmatization to input text 
    ---
    Parameters:
        text: the text to be lemmatized
    ---
    Returns:
        text: text that has been lemmatized
    """
    #create lemmatize object
    wnl = WordNetLemmatizer()

    lemmas = [wnl.lemmatize(word) for word in text.split()]
    text = ' '.join(lemmas)

    return text

def remove_stopwords(text, extra_words=None, exclude_words=None):
    """ 
    Purpose:
        to remove stopwords from input text 
    ---
    Parameters:
        text: text from which to remove stop words
    ---
    Returns:
        text: text that has had stopwords removed
    """

    stopwords_list = stopwords.words('english')

    if extra_words != None:
        stopwords_list.extend(extra_words)

    if exclude_words != None:
        for w in exclude_words:
            stopwords_list.remove(w)

    words = text.split()

    filtered_words = [w for w in words if w not in stopwords_list]

    print('Removed {} stopwords'.format(len(words) - len(filtered_words)))
    print('---')

    text = ' '.join(filtered_words)

    return text

def clean(text, extra_words=None, exclude_words=None):
    """ 
    Purpose:
        performs basic clean, tokenization, and removal of stopwords on input text
    ---
    Parameters:
        text
        extra_words
        exclude_words
    ---
    Returns:
        text
    """

    text = basic_clean(text)
    
    text = tokenize(text)

    text = remove_stopwords(text, extra_words, exclude_words)

    return text

def prep_text(df):
    """ 
    Purpose:
        
    ---
    Parameters:
        
    ---
    Returns:
    
    """
    df['clean'] = df.original.apply(clean)
    df['stemmed'] = df.clean.apply(stem)
    df['lemmatized'] = df.clean.apply(lemmatize)

    return df