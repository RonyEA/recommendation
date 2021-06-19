from flask import Flask, render_template, request
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer

from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

import time 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
seed = 8

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score


from sklearn.naive_bayes import MultinomialNB, BernoulliNB

from imblearn.over_sampling import SMOTE
import joblib

import xgboost as xgb


app = Flask(__name__)


# df = pd.read_csv("sample30.csv")
# fin_ratings = pd.read_csv("fin_ratings.csv")
# fin_ratings = fin_ratings.set_index('userId')
# lis = fin_ratings.loc["robin"].sort_values(ascending=False).index[:20]


# df_recom = df[df['name'].isin(lis)]



def cleantext(df): 
    ### dont change the original tweet
    # remove emoticons form the tweets
    words_remove = ["ax","i","you","edu","s","t","m","subject","can","lines","re","what", "there","all","we",
                    "one","the","a","an","of","or","in","for","by","on","but","is","in","a","not","with","as",
                    "was","if","they","are","this","and","it","have","has","from","at","my","be","by","not","that","to",
                    "from","com","org","like","likes","so","said","from","what","told","over","more","other",
                    "have","last","with","this","that","such","when","been","says","will","also","where","why",
                    "would","today", "in", "on", "you", "r", "d", "u", "hw","wat", "oly", "s", "b", "ht", 
                    "rt", "p","the","th", "n", "was"]

    df['reviews_text'] = df['reviews_text'].replace(r'<ed>','', regex = True)
    df['reviews_text'] = df['reviews_text'].replace(r'\B<U+.*>|<U+.*>\B|<U+.*>','', regex = True)
    
    # convert tweets to lowercase
    df['reviews_text'] = df['reviews_text'].str.lower()
    
    #remove user mentions
    df['reviews_text'] = df['reviews_text'].replace(r'^(@\w+)',"", regex=True)
    
    #remove_symbols
    df['reviews_text'] = df['reviews_text'].replace(r'[^a-zA-Z0-9]', " ", regex=True)

    #remove punctuations 
    df['reviews_text'] = df['reviews_text'].replace(r'[[]!"#$%\'()\*+,-./:;<=>?^_`{|}]+',"", regex = True)
    
    #remove words of length 1 or 2 
    df['reviews_text'] = df['reviews_text'].replace(r'\b[a-zA-Z]{1,2}\b','', regex=True)

    #remove extra spaces in the tweet
    df['reviews_text'] = df['reviews_text'].replace(r'^\s+|\s+$'," ", regex=True)
     
    #remove stopwords and words_to_remove
    stop_words = set(stopwords.words('english'))
    mystopwords = [stop_words, "via", words_remove]
    
    df['reviews_text'] = df['reviews_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in mystopwords]))

    df = df.replace({"Positive":1, "Negative":0})
    
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english',
        ngram_range=(1, 3), # try 1,3
        max_features=50000
        )
    all_text = df['reviews_text']
    word_vectorizer.fit(all_text)
    train_word_features = word_vectorizer.transform(all_text)

    model = joblib.load("joblib_model_xgb.pkl")
    preds = model.predict(train_word_features)
    df['preds'] = preds
    
    return df


df = pd.read_csv("sample30.csv")
fin_ratings = pd.read_csv("./Data/fin_ratings.csv")
fin_ratings = fin_ratings.set_index('userId')
# 


# df_recom = df[df['name'].isin(lis)]



# fin_df = fin_df[['name', 'preds']]
# fin_df.groupby('name').mean().sort_values(ascending=False, by="preds")*100

# df = pd.read_csv("./Data/fin_ratings.csv")
# df = df.set_index('userId')


@app.route("/", methods=['POST', "GET"])
def home():
    if request.method == 'POST':
        user = request.form['nm']
        if user in fin_ratings.index.tolist():
            lis = fin_ratings.loc[user].sort_values(ascending=False).index[:20]
            df_recom = df[df['name'].isin(lis)]
            fin_df = cleantext(df_recom)
            fin_df = fin_df[['name', 'preds']]
            d = fin_df.groupby('name').mean().sort_values(ascending=False, by="preds")*100
            # d = df.loc[user].sort_values(ascending=False)
            products = d[:5].index.tolist()
            return render_template('index.html', products=products, submit="yes")
        else:
            return render_template('index.html', products="None")
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)