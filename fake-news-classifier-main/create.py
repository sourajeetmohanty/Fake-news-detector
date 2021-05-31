# -*- coding: utf-8 -*-
"""Fake News.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mkx5XHT1KvshAhtIcsC8LNe_dP7du9BQ

### Data Import
"""

import numpy as np
import pandas as pd

True_news = pd.read_csv('True.csv')
Fake_news = pd.read_csv('Fake.csv')

True_news['label'] = 0

Fake_news['label'] = 1

True_news.head()

Fake_news.head()

dataset1 = True_news[['text','label']]
dataset2 = Fake_news[['text','label']]

dataset = pd.concat([dataset1 , dataset2])

dataset.shape

"""### Null values"""

dataset.isnull().sum() # no null values

"""### Balanced or Unbalanced dataset"""

dataset['label'].value_counts()

dataset1.shape # true news

dataset2.shape # fake news

"""### Shuffle or Resample"""

dataset = dataset.sample(frac = 1)

dataset.head(20)

import nltk

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')

ps = WordNetLemmatizer()

stopwords = stopwords.words('english')

nltk.download('wordnet')

def cleaning_data(row):
    
    # convert text to into lower case
    row = row.lower() 
    
    # this line of code only take words from text and remove number and special character using RegX
    row = re.sub('[^a-zA-Z]' , ' ' , row)
    
    # split the data and make token.
    token = row.split() 
    
    # lemmatize the word and remove stop words like a, an , the , is ,are ...
    news = [ps.lemmatize(word) for word in token if not word in stopwords]  
    
    # finaly join all the token with space
    cleanned_news = ' '.join(news) 
    
    # return cleanned data
    return cleanned_news

dataset['text'] = dataset['text'].apply(lambda x : cleaning_data(x))

dataset.isnull().sum()

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features = 50000 , lowercase=False , ngram_range=(1,2))

dataset.shape

X = dataset.iloc[:35000,0]
y = dataset.iloc[:35000,1]

X.head()

y.head()

from sklearn.model_selection import train_test_split
train_data , test_data , train_label , test_label = train_test_split(X , y , test_size = 0.2 ,random_state = 0)

vec_train_data = vectorizer.fit_transform(train_data)

vec_train_data = vec_train_data.toarray()

train_data.shape , test_data.shape

vec_test_data = vectorizer.transform(test_data).toarray()

vec_train_data.shape , vec_test_data.shape

train_label.value_counts() # balanced partition

test_label.value_counts() # balanced partition

training_data = pd.DataFrame(vec_train_data , columns=vectorizer.get_feature_names())
testing_data = pd.DataFrame(vec_test_data , columns= vectorizer.get_feature_names())

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score,classification_report

clf = MultinomialNB()

clf.fit(training_data, train_label)
y_pred  = clf.predict(testing_data)

"""### MultinomialNB"""

pd.Series(y_pred).value_counts()

test_label.value_counts()

print(classification_report(test_label , y_pred))

"""Now predict on both train set"""

y_pred_train = clf.predict(training_data)
print(classification_report(train_label , y_pred_train))

accuracy_score(train_label , y_pred_train)

accuracy_score(test_label , y_pred)

news = cleaning_data(str("Ivanka Trump marries Donald Trump."))

single_prediction = clf.predict(vectorizer.transform([news]).toarray())
single_prediction

"""### Save the Model"""

import joblib

joblib.dump(clf , 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

model = joblib.load('model.pkl')