# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import random
import pandas as pd
from datetime import datetime
from common import utils

#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import traceback

import numpy as np
import pandas as pd
import sagemaker as sage
from time import gmtime, strftime
import numpy as np

pd.options.mode.chained_assignment = None
from copy import deepcopy
from string import punctuation
from random import shuffle

import re
from string import punctuation 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import boto3
import json
import pandas as pd
from sagemaker import get_execution_role
from io import StringIO
import datetime

import keras
import tensorflow
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Activation
from keras.layers.embeddings import Embedding
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

import emoji

# Optional
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# These are the paths to where SageMaker mounts interesting things in your
# container.
prefix = '/opt/ml/'

input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')

# This algorithm has a single channel of input data called 'training'.
# Since we run in File mode, the input files are copied to the directory
# specified here.
channel_name = 'training'
training_path = os.path.join(input_path, channel_name)

max_length = 0
vocab_size = 0
EMBEDDING_DIM = 100
   
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
                   "can't've": "cannot have", "'cause": "because", "could've": "could have", 
                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
                   "he'll've": "he will have", "he's": "he is", "how'd": "how did", 
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
                   "I'll've": "I will have","I'm": "I am", "I've": "I have", 
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
                   "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
                   "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                   "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                   "she's": "she is", "should've": "should have", "shouldn't": "should not", 
                   "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                   "this's": "this is",
                   "that'd": "that would", "that'd've": "that would have","that's": "that is", 
                   "there'd": "there would", "there'd've": "there would have","there's": "there is", 
                       "here's": "here is",
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                   "they'll've": "they will have", "they're": "they are", "they've": "they have", 
                   "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                   "we're": "we are", "we've": "we have", "weren't": "were not", 
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
                   "what's": "what is", "what've": "what have", "when's": "when is", 
                   "when've": "when have", "where'd": "where did", "where's": "where is", 
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                   "who's": "who is", "who've": "who have", "why's": "why is", 
                   "why've": "why have", "will've": "will have", "won't": "will not", 
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                   "you'll've": "you will have", "you're": "you are", "you've": "you have" } 

smileys ={
        ":‑)":"smiley",
        ":-]":"smiley",
        ":-3":"smiley",
        ":->":"smiley",
        "8-)":"smiley",
        ":-}":"smiley",
        ":)":"smiley",
        ":]":"smiley",
        ":3":"smiley",
        ":>":"smiley",
        "8)":"smiley",
        ":}":"smiley",
        ":o)":"smiley",
        ":c)":"smiley",
        ":^)":"smiley",
        "=]":"smiley",
        "=)":"smiley",
        ":-))":"smiley",
        ":‑D":"smiley",
        "8‑D":"smiley",
        "x‑D":"smiley",
        "X‑D":"smiley",
        ":D":"smiley",
        "8D":"smiley",
        "xD":"smiley",
        "XD":"smiley",
        ":‑(":"sad",
        ":‑c":"sad",
        ":‑<":"sad",
        ":‑[":"sad",
        ":(":"sad",
        ":c":"sad",
        ":<":"sad",
        ":[":"sad",
        ":-||":"sad",
        ">:[":"sad",
        ":{":"sad",
        ":@":"sad",
        ">:(":"sad",
        ":'‑(":"sad",
        ":'(":"sad",
        ":‑P":"playful",
        "X‑P":"playful",
        "x‑p":"playful",
        ":‑p":"playful",
        ":‑Þ":"playful",
        ":‑þ":"playful",
        ":‑b":"playful",
        ":P":"playful",
        "XP":"playful",
        "xp":"playful",
        ":p":"playful",
        ":Þ":"playful",
        ":þ":"playful",
        ":b":"playful",
        "<3":"love"
        }
        
def clean_tokens(tweet):

    tweet = tweet.lower()
    
    tokens = tokenizer.tokenize(tweet)
    
    #remove call outs
    tokens = filter(lambda t: not t.startswith('@'), tokens)
    
    tweet = " ".join(tokens)
    
    # convert emojis to words
    tweet = emoji.demojize(tweet).replace(":"," ").replace("_"," ")
    #remove numbers
    tweet = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)', '', tweet)
    tweet = re.sub(r'/([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)/', '', tweet)
    #clean apostrophe
    tweet = tweet.replace("’","'")
    tweet = tweet.replace('“','"')
    tweet = tweet.replace('”','"')
    tweet = tweet.replace('…','')
    tweet = tweet.replace('\n','')
    tweet = tweet.replace('...','')
    tweet = tweet.replace('..','')
    tweet = tweet.replace('�','')
    tweet = tweet.replace('£','')
    tweet = tweet.replace('·','')
    tweet = tweet.replace('–','')
    tweet = tweet.replace('🏻','')
    tweet = tweet.replace('à','')
    tweet = tweet.replace(' ‍','')
    
    tokens = tokenizer.tokenize(tweet)
    
    #remove hashtags
    tokens = filter(lambda t: not t.startswith('#'), tokens)
    #remove urls
    tokens = filter(lambda t: not t.startswith('http'), tokens)
    tokens = filter(lambda t: not t.startswith('t.co/'), tokens)
    tokens = filter(lambda t: not t.startswith('ow.ly/'), tokens)
    tokens = filter(lambda t: not t.startswith('bit.ly/'), tokens)
    tokens = filter(lambda t: not t.startswith('soundcloud.com/'), tokens)
    tokens = filter(lambda t: not t.startswith('outline.com/'), tokens)
    
    new_tokens = []
    for token in tokens:
        if len(token.strip())>0:
            new_tokens.append(token)
    
    _stopwords = set(list(punctuation) + ['AT_USER','URL'])
    #_stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])
    
    new_tokens = [smileys[word] if word in smileys else word for word in new_tokens]
    new_tokens = [contraction_mapping[word] if word in contraction_mapping else word for word in new_tokens]
    new_tokens = [word for word in new_tokens if word not in _stopwords]
    
    return new_tokens
    
# Process and prepare the data
def data_process(raw_data):
    
    raw_data=raw_data[raw_data['sentiment']!='MIXED']

    raw_data['tweet'] = raw_data['tweet'].apply(lambda x: clean_tokens(str(x)))

    x_train = raw_data['tweet']
    y_train = raw_data['sentiment']

    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_train_Y = encoder.transform(y_train)
    dummy_train_y = np_utils.to_categorical(encoded_train_Y)
    
    tokenizer_obj = Tokenizer()   

    tokenizer_obj.fit_on_texts(x_train)    
    max_length = 100 # max([len(s.split()) for s in x_train])    
    vocab_size = len(tokenizer_obj.word_index)+1  

    #Building the vectors of words
    x_train_tokens = tokenizer_obj.texts_to_sequences(x_train)

    x_train_pad = pad_sequences(x_train_tokens, maxlen=max_length, padding='post')
    
    return x_train_pad, dummy_train_y, vocab_size, max_length
    
# Building the ANN
def baseline_model(vocab_size, max_length):

    # create model
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, kernel_initializer="normal", activation='softmax'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
    
def generate_model(X_train, y_train, vocab_size, max_length):

    estimator = baseline_model(vocab_size, max_length)
    estimator.fit(X_train, y_train, epochs=25, batch_size=128, verbose=2)
    return estimator


def main(argv=None):
    parser = argparse.ArgumentParser(description='SageMaker Training Job')
    parser.add_argument('--s3_training_data', type=str, help='Training Dataset.')
    parser.add_argument('--s3_training_predictions', type=str, help='Location to place training results.')
    parser.add_argument('--s3_model_artifacts', type=str, help='Location for Model Artifacts')
    parser.add_argument('--model_name', type=str, help='Location for Model Artifacts')
    parser.add_argument('--max_length', type=str, help='Location for Model Artifacts')
    parser.add_argument('--vocab_size', type=str, help='Location for Model Artifacts')
    args = parser.parse_args()

    training_data_file = utils.s3_get_file(args.s3_training_data)

    print('Starting the training.')
    try:
        
        # read in training data to pandas dataframe.
        raw_data = pd.read_csv(training_data_file)

        y_train = raw_data['sentiment']
        
        # one hot encode the sentiment
        encoder = LabelEncoder()
        encoder.fit(y_train)
        encoded_train_y = encoder.transform(y_train)
        dummy_train_y = np_utils.to_categorical(encoded_train_y)
    
        collist = raw_data.columns.tolist()
        collist.remove('sentiment')
        x_train = raw_data[collist]
        
        # create keras classifier and fit the model
        optimized_classifier = generate_model(x_train, dummy_train_y, args.vocab_size, args.max_length)
        
        # make predictions with training data
        predictions = optimized_classifier.predict(y_train)
        
        #write predictions to file system
        train.to_csv(os.path.join(model_path, 'predictions.csv'), sep=',')
        
        # save the model to the hard drive
        optimized_classifier.model.save(os.path.join(model_path, args.model_name))
        
        # upload model to s3
        utils.s3_upload_file(args.s3_model_artifacts, os.path.join(model_path, args.model_name))
        
        # upload training predictions
        utils.s3_upload_file('{}/predictions.csv'.format(args.s3_training_predictions), os.path.join(model_path, 'predictions.csv'))
        
        with open('/tmp/s3_training_predictions.txt', 'w') as f:
            f.write('{}/predictions.csv'.format(s3_training_predictions))
            
        with open('/tmp/s3_model_artifacts.txt', 'w') as f:
            f.write('{}/model.tar.z'.format(s3_model_artifacts))
            
        print('Training is complete.')
        
    except Exception as e:
        
        # Write out an error file. This will be returned as the failure
        # Reason in the DescribeTrainingJob result.
        trc = traceback.format_exc()
        
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        
        # Printing this causes the exception to be in the training job logs
        print(
            'Exception during training: ' + str(e) + '\n' + trc,
            file=sys.stderr)
        
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)

if __name__== "__main__":
  main()