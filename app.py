from flask import Flask, render_template, request
import json
import tensorflow as tf

# Imports and code from main.py
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn import preprocessing

# Load the data and tokenizer from the main.py code
data = pd.read_csv("news.csv")
data = data.drop(["Unnamed: 0"], axis=1)
le = preprocessing.LabelEncoder()
le.fit(data['label'])
data['label'] = le.transform(data['label'])
tokenizer1 = Tokenizer()
tokenizer1.fit_on_texts(data['title'])

# Load the trained model
model = load_model('news_model.h5')

# Function to detect fake news
def detect_fake_news(model, tokenizer, text):
    sequences = tokenizer.texts_to_sequences([text])[0]
    padded_sequences = pad_sequences([sequences], maxlen=54, padding='post', truncating='post')
    prediction = model.predict(padded_sequences)
    return 'Real news' if prediction >= 0.5 else 'Fake news'

# Flask application
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    text = request.form['text']
    result = detect_fake_news(model, tokenizer1, text)
    return json.dumps({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
