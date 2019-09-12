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


import datetime
import os
import subprocess
import time

import boto3
from botocore.exceptions import ClientError
from botocore.utils import InstanceMetadataFetcher
from botocore.credentials import InstanceMetadataProvider

import json
from common.S3Url import S3Url
from pyathena import connect
import pandas as pd

def get_creds():
    provider = InstanceMetadataProvider(iam_role_fetcher=InstanceMetadataFetcher(timeout=1000, num_attempts=2))
    creds = provider.load()
    return creds;

def get_session():
    creds = get_creds()
    session = boto3.Session(
        aws_access_key_id = creds.access_key,
        aws_secret_access_key = creds.secret_key,
        aws_session_id = creds.token)
    return session

def s3_get_file(file_name, input_path):
    
    s3f = S3Url(file_name)
    s3 = get_session().client('s3')
    
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    
    s3.download_file(s3f.bucket, s3f.key, '{}/{}'.format(input_path, s3f.file_name))
    
    return '{}/{}'.format(input_path, s3f.file_name)

def encrypt_file(file_name, kms_key):
    
    return "Success"


def decrypt_file(file_name, kms_key):
    
    return "Success"
    
def s3_upload_file(s3_location, local_file):

    s3 = get_session().client('s3')
    
    s3f = S3Url(s3_location)
    
    s3.upload_file(local_file, s3f.bucket, s3f.key)

    
    
def athena_get_file():

    creds = get_creds()
    
    conn = connect(
        aws_access_key_id = creds.access_key,
        aws_secret_access_key = creds.secret_key,
        s3_staging_dir='s3://YOUR_S3_BUCKET/path/to/',
        region_name='us-west-2')
    
    df = pd.read_sql("SELECT * FROM many_rows", conn)
    