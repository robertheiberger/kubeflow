#!/usr/bin/env python3

import kfp
from kfp import components
from kfp import dsl
from kfp.aws import use_aws_secret
from datetime import datetime

train_op = components.load_component_from_file('components/train/component.yaml')
test_op = components.load_component_from_file('components/test/component.yaml')
preprocess_op = components.load_component_from_file('components/preprocess/component.yaml')

@dsl.pipeline(
    name='Twitter NNet Classification pipeline',
    description='Twitter Classification using NNet in Kubeflow'
)
def twitter_classification(
    s3_raw_data = 's3://kubeflow-meda/data/raw/tweets.csv',
    s3_model_data = 's3://kubeflow-meda/models',
    model_name = 'NNET'
    ):
    
    # preprocess data. cleansing and feature engineering. Also creating s3 folder structure to 
    # store data and artifacts of the model run.
    preprocess = preprocess_op(
        s3_raw_data = s3_raw_data,
        model_name = model_name
    ).apply(use_aws_secret('aws-secret', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'))
    
    training = train_op(
        s3_training_data = preprocess.outputs['s3_training_data'],
        s3_training_predictions = preprocess.outputs['s3_training_predictions'],
        s3_model_artifacts = preprocess.outputs['s3_model_artifacts'],
        model_name = model_name,
        max_length = preprocess.outputs['max_length'],
        vocab_size = preprocess.outputs['vocab_size']
    ).apply(use_aws_secret('aws-secret', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'))

    testing = test_op(
        s3_testing_data = preprocess.outputs['s3_testing_data'],
        s3_testing_predictions = preprocess.outputs['s3_testing_predictions'],
        s3_model_artifacts = training.outputs['s3_model_artifacts'],
        model_name = model_name,
        max_length = preprocess.outputs['max_length'],
        vocab_size = preprocess.outputs['vocab_size']
    ).apply(use_aws_secret('aws-secret', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'))

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(mnist_classification, __file__ + '.zip')