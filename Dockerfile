FROM python:3.9

RUN apt-get install git

WORKDIR /ML

RUN git clone https://github.com/wmcnally/kapao.git

WORKDIR /ML/kapao

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN python -m pip install -r requirements.txt
RUN python -m pip install flask black


RUN  python data/scripts/download_models.py