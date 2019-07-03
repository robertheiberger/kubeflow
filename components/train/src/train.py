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
#!/usr/bin/env python

from __future__ import print_function

import argparse
import logging
import random
import pandas as pd
from datetime import datetime
from common import utils


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

# Building the ANN
def baseline_model(vocab_size, max_length):

    # create model
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length = max_length))
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
    parser.add_argument('--max_length', type=int, help='Location for Model Artifacts')
    parser.add_argument('--vocab_size', type=int, help='Location for Model Artifacts')
    args = parser.parse_args()
    
    training_data_file = utils.s3_get_file(args.s3_training_data, input_path)

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
    
        print(dummy_train_y)
        
        collist = raw_data.columns.tolist()
        collist.remove('sentiment')
        x_train = raw_data[collist]
        
        print(x_train)
        
        # create keras classifier and fit the model
        optimized_classifier = generate_model(x_train, dummy_train_y, args.vocab_size, args.max_length)
        
        # make predictions with training data
        predictions = pd.DataFrame(optimized_classifier.predict(x_train))
        
        predictions = pd.concat([y_train, predictions], axis=1, join='inner')
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        #write predictions to file system
        predictions.to_csv(os.path.join(output_path, 'predictions.csv'), sep=',')
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        # save the model to the hard drive
        optimized_classifier.model.save(os.path.join(model_path, '{}.h5'.format(args.model_name)))
              
        # upload model to s3
        utils.s3_upload_file('{}/{}.h5'.format(args.s3_model_artifacts, args.model_name), 
                             os.path.join(model_path, '{}.h5'.format(args.model_name)))
        
        # upload training predictions
        utils.s3_upload_file('{}/predictions.csv'.format(args.s3_training_predictions), 
                             os.path.join(output_path, 'predictions.csv'))
        
        with open('/tmp/s3_training_predictions.txt', 'w') as f:
            f.write('{}/predictions.csv'.format(args.s3_training_predictions))
            
        with open('/tmp/s3_model_artifacts.txt', 'w') as f:
            f.write('{}/{}.h5'.format(args.s3_model_artifacts, args.model_name))
            
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
