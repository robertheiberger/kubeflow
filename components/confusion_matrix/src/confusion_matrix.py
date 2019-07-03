# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# A program to generate confusion matrix data out of prediction results.
# Usage:
# python confusion_matrix.py  \
#   --predictions=gs://bradley-playground/sfpd/predictions/part-* \
#   --output=gs://bradley-playground/sfpd/cm/ \
#   --target=resolution \
#   --analysis=gs://bradley-playground/sfpd/analysis \

from __future__ import print_function

import argparse
import json
import os
import urlparse
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.python.lib.io import file_io
from sklearn.preprocessing import LabelEncoder
from common import utils

prefix = '/opt/ml/'

input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')

def main(argv=None):
    parser = argparse.ArgumentParser(description='Confusion Matrix Job')
    parser.add_argument('--s3_targets', type=str, help='Training Dataset.')
    parser.add_argument('--s3_predictions', type=str, help='Location to place training results.')
    args = parser.parse_args()
    
    try:
        targets_data_file = utils.s3_get_file(args.s3_testing_data, input_path)
        predictions_data_file = utils.s3_get_file(args.s3_testing_data, input_path)

        target_data = pd.read_csv(targets_data_file, index = False)
        prediction_data = pd.read_csv(predictions_data_file, index = False)
        
        target_sentiment = target_data['sentiment']
        
        encoder = LabelEncoder()
        encoder.fit(target_sentiment)
        encoded_target_sentiment = encoder.transform(target_sentiment)
        dummy_encoded_target_sentiment = np_utils.to_categorical(encoded_target_sentiment)
        
        predicted_sentiment = encoder.inverse_transform(prediction_data)
        
        #classes = np.array(['NEGATIVE', 'NEUTRAL', 'POSITIVE']
        classes = list(target_sentiment.unique())
        cm = confusion_matrix(target_sentiment, predicted_sentiment, labels = classes)
  
        data = []
        for target_index, target_row in enumerate(cm):
            for predicted_index, count in enumerate(target_row):
                data.append((classes[target_index], classes[predicted_index], count))

        df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])
        cm_file = os.path.join(output_path, 'confusion_matrix.csv')
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        with file_io.FileIO(cm_file, 'w') as f:
            df_cm.to_csv(f, columns=['target', 'predicted', 'count'], header=False, index=False)

        metadata = {
            'outputs' : [{
            'type': 'confusion_matrix',
            'format': 'csv',
            'schema': [
                {'name': 'target', 'type': 'CATEGORY'},
                {'name': 'predicted', 'type': 'CATEGORY'},
                {'name': 'count', 'type': 'NUMBER'},
            ],
            'source': cm_file,
            # Convert vocab to string because for boolean values we want "True|False" to match csv data.
            'labels': list(map(str, vocab)),
            }]
          }

        with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as f:
            json.dump(metadata, f)

        accuracy = accuracy_score(df['target'], df['predicted'])
  
        metrics = {
            'metrics': [{
            'name': 'accuracy-score',
            'numberValue':  accuracy,
            'format': "PERCENTAGE",
            }]
        }
  
        with file_io.FileIO('/mlpipeline-metrics.json', 'w') as f:
            json.dump(metrics, f)
                           
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
