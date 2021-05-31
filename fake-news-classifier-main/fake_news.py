import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

ps = WordNetLemmatizer()
stopwords = stopwords.words('english')
vectorizer = joblib.load('vectorizer.pkl')
true_dict = joblib.load('true_dict.pkl')
false_dict = joblib.load('false_dict.pkl')

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




clf = joblib.load('model.pkl')

from flask import Flask, render_template, request
import time

app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def hello_world():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        news = cleaning_data(str(request.form['article']))
        start = time.time()
        wordvec = vectorizer.transform([news]).toarray()
        proba = clf.predict_proba(wordvec)
        print('proba', proba)
        single_prediction = clf.predict(wordvec)
        end = time.time()
        print("Predicted", single_prediction, "in", end - start, "seconds from", news)
        if single_prediction[0] == 0:
            result = True
        else:
            result = False

        true_plot, false_plot = get_wordclouds(news)
        return render_template('result.html', result=result, true_plot=true_plot, false_plot=false_plot, truth_prob=proba[0][0]*100, fake_prob=proba[0][1]*100)

def get_wordclouds(news):
    words = news.split()
    true_vocab = {}
    false_vocab = {}
    for word in words:
        true_vocab[word] = true_dict.get(word, 0)
        false_vocab[word] = false_dict.get(word,0)
    
    true_wc = WordCloud(width=640, height=480).generate_from_frequencies(true_vocab)
    plt.figure()
    plt.imshow(true_wc, interpolation="bilinear")
    plt.title("True News Wordcloud")
    plt.axis("off")

    true_file = BytesIO()
    plt.savefig(true_file, format='png')
    true_file.seek(0)
    true_data = base64.b64encode(true_file.getvalue())

    false_wc = WordCloud(width=640, height=480).generate_from_frequencies(false_vocab)
    plt.figure()
    plt.imshow(false_wc, interpolation="bilinear")
    plt.title("Fake News Wordcloud")
    plt.axis("off")

    false_file = BytesIO()
    plt.savefig(false_file, format='png')
    false_file.seek(0)
    false_data = base64.b64encode(false_file.getvalue())

    return (true_data.decode('utf-8'), false_data.decode('utf-8'))
