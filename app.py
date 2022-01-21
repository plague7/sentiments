import os
import os.path

import streamlit as st
from tensorflow import keras
from tensorflow.keras.utils import *
import tensorflow as tf
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalAvgPool1D

#LOADING MODEL#
#Loading the model#

#Test#
#model = load_model("C:/Users/Simplon/Desktop/Travaux python/Texte analyze/Sentiment analyses/Sentiment analyses deploying/matthis_model.h5", compile=False)

#When deploying#
MODEL_DIR = os.path.join(os.path.dirname('__file__'), 'matthis_model.h5')
model = keras.models.load_model(MODEL_DIR)

#STOPWORDS#
stop_words = set(stopwords.words('english'))
stop_words.remove('not')

#TOKENIZER#
#with open("C:/Users/Simplon/Desktop/Travaux python/Texte analyze/Sentiment analyses/Sentiment analyses deploying/tokenizer.pickle" , 'rb') as handle:
#    tokenizer = pickle.load(handle)
TOKENIZER_DIR = os.path.join(os.path.dirname('__file__'), 'tokenizer.pickle')
with open(TOKENIZER_DIR , 'rb') as handle:
    tokenizer = pickle.load(handle)

#Some random shit#
maxlen = int(50)
#Padded_train = pad_sequences(Tokenized_train, maxlen=maxlen, padding='pre')
#Padded_val = pad_sequences(Tokenized_val, maxlen=maxlen, padding='pre')

#FUNCTIONS#

#Remove punctuations#
def remove_punctuations_numbers(inputs):
    return re.sub(r'[^a-zA-Z]', ' ', inputs)

#Tokenization of the input#
def tokenization(inputs):  # Ref.1
    return word_tokenize(inputs)

#Removing stopwords#
def stopwords_remove(inputs):  # Ref.2
    return [k for k in inputs if k not in stop_words]

#Lemmatizer#
def lemmatization(inputs):  # Ref.1
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word=kk, pos='v') for kk in inputs]

#Removing small words (less than 2)#
def remove_less_than_2(inputs):  # Ref.1
    inputs = [j for j in inputs if len(j) > 2]

#Predict the written review#
def predict_recommendation(input_text):  # The function for doing all the previous steps
    input_text = input_text.lower()
    input_text = re.sub(r'[^a-zA-Z]', ' ', input_text)
    input_text = tokenization(input_text)
    input_text = stopwords_remove(input_text)
    input_text = lemmatization(input_text)
    input_text = ' '.join(input_text)
    input_text = tokenizer.texts_to_sequences([input_text])
    input_text = pad_sequences(input_text, maxlen=maxlen, padding='pre')
    prediction = model.predict(input_text)
    if prediction >= 0.5:
        st.write('It seems to be a good review !')
        st.write(f'Recommended with %{round(float(prediction*100), 2)}')
    else:
        st.write('IT may be a bad review')
        st.write(f'Not Recommended with %{round(float(prediction*100), 2)}')


#STREAMLIT#

#STREAMLIT FUNCTIONS#

def user_input():
    st.write('You will have to write down a complete random review of some clothing for woman you could find online, in any website.')
    st.write('Then, the app will try to detect if you are giving it a good review, or not')
    text = str(st.text_input('Type your review here !'))
    return text

#Header#
st.title('Welcome to our online sentiment analysis app!')
st.header('Please select one specific use for our app:')

### Selectbox ###

### Randomizing Tool View ###
if st.selectbox('Different analysis models', ['Women online clothing',
                 'Sarcasm detector']) == 'Women online clothing':
    ### Entering the review yourself###
    text=user_input()
    if st.button('Am I giving it a good review, or a bad one ?'):
        #input_analysis(text)
        predict_recommendation(text)
else:
    st.write('Work in progress')
    