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
#FROM ubuntu:16.04
FROM python:3.6
#RUN apt-get update -y && apt-get install --no-install-recommends -y -q ca-certificates python-dev python-setuptools wget unzip

RUN easy_install pip

RUN pip install boto3==1.9.169 sagemaker pathlib2
RUN pip install numpy
RUN pip install scipy
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install flask
RUN pip install gevent
RUN pip install gunicorn
RUN pip install tensorflow==1.13.1
RUN pip install keras==2.2.4
RUN pip install gensim
RUN pip install tqdm
RUN pip install nltk
RUN pip install emoji
RUN pip install sagemaker

COPY train/src/train.py .
COPY test/src/test.py .
COPY preprocess/src/preprocess.py .
COPY confusion_matrix/src/confusion_matrix.py .

COPY common/utils.py /app/common/
COPY common/S3Url.py /app/common/

ENV PYTHONPATH /app

ENTRYPOINT [ "bash" ]
