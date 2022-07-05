FROM python:3.9

RUN apt-get install git

WORKDIR /ML

RUN git clone https://github.com/wmcnally/kapao.git

WORKDIR /ML/kapao

RUN python -m pip install -r requirements.txt
RUN python -m pip install flask


RUN  python data/scripts/download_models.py