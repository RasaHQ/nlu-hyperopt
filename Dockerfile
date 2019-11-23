FROM rasa/rasa:latest-full

COPY requirements.txt .
COPY data /data
COPY nlu_hyperopt /nlu_hyperopt
COPY scripts /scripts

RUN pip install -r requirements.txt && chmod -R +x /scripts

WORKDIR "/"
