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
    image='174872318107.dkr.ecr.us-west-2.amazonaws.com/kmeans:1',
    s3RawData='s3://kubeflow-pipeline-meda/data/raw/tweets.csv',
    s3ModelData='s3://kubeflow-pipeline-meda/models',
    ModelName = 'NNET',
    role_arn=''
    ):
    
    now = datetime.datetime.now()
    
    # marking a unique model run
    s3ModelRunId = "{}_{}{}{}{}{}{}".format(modelName, a.year, a.month, a.day, a.hour, a.minute, a.second)
    
    # setting folder structure
    s3TrainingPredictions = "{}/{}/data/train/pred".format(s3ModelData, s3ModelRunId)
    s3TestingPredictions = "{}/{}/data/test/pred".format(s3ModelData, s3ModelRunId)
    s3TrainingData = "{}/{}/data/train/input".format(s3ModelData, s3ModelRunId)
    s3TestingData = "{}/{}/data/test/input".format(s3ModelData, s3ModelRunId)
    s3ModelArtifacts = "{}/{}/artifacts".format(s3ModelData, s3ModelRunId)
    
    # preprocess data. cleansing and feature engineering
    preprocess = preprocess_op(
        s3RawData=s3RawData,
        s3TrainingData=s3TrainingData,
        s3TestingData=s3TestingData,
    ).apply(use_aws_secret('aws-secret', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'))

    training = train_op(
        s3TrainingData="{}/{}".format(s3TrainingData, 'tweets.csv'),
        s3TrainingPredictions="{}/{}".format(s3TestingData, 'tweets.csv'),
        s3ModelArtifacts = s3ModelArtifacts,
        ModelName = ModelName
    ).apply(use_aws_secret('aws-secret', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'))

    testing = test_op(
        region=region,
        image=image,
        model_artifact_url=training.outputs['model_artifact_url'],
        model_name=training.outputs['job_name'],
        role=role_arn
    ).apply(use_aws_secret('aws-secret', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'))

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(mnist_classification, __file__ + '.zip')