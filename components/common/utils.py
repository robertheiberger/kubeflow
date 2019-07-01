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
import json
import S3Url

def s3_get_file(file_name, input_path):
    
    s3f = S3Url(file_name)
    s3 = boto3.client('s3', 
                      aws_access_key_id='AKIAQARR2PO37LRUVN3Z', 
                      aws_secret_key_id='AM1x8zMZZe97doETi7UcK2jT6vqdNNjAWBalkWmN')
    s3.download_file(s3f.bucket, s3f.key, '{}/{}'.format(input_path, s3f.file_name))
    
    return '{}/{}'.format(input_path, s3f.file_name)

def encrypt_file(file_name, kms_key):
    
    return "Success"


def decrypt_file(file_name, kms_key):
    
    return "Success"
    
def s3_upload_file(s3_location, local_file):

    s3 = boto3.client('s3', 
                      aws_access_key_id='AKIAQARR2PO37LRUVN3Z', 
                      aws_secret_key_id='AM1x8zMZZe97doETi7UcK2jT6vqdNNjAWBalkWmN')
    
    s3f = S3Url(file_name)
    
    s3.client.upload_file(local_file, s3f.bucket, s3f.key)

    
    