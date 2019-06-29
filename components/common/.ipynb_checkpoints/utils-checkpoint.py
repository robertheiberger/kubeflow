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

def s3_get_file(file_name):
    
    s3f = S3Url(file_name)
    s3 = boto3.resource('s3')
    s3.download_file(s3f.bucket, s3f.key, s3f.file_name)
    return s3f.file_name
    
def s3_get_test_file(test_file):

    return s3_get_file(test_file)

def get_train_file(train_file):
    
    return s3_get_file(train_file)

def encrypt_file(file_name, kms_key):
    
    return "Success"


def decrypt_file(file_name, kms_key):
    
    return "Success"
    
def s3_upload_file(bucket, s3folder, file_name):

    s3 = boto3.resource('s3')
    s3.meta.client.upload_file(file_name, bucket, '{}/{}'.format(s3folder, file_name))

    
    