FROM rasa/rasa:1.4.5-full

COPY requirements.txt .
COPY data /data
COPY nlu_hyperopt /nlu_hyperopt
COPY scripts /scripts

RUN pip install -U pip && pip install -r requirements.txt && chmod -R +x /scripts

WORKDIR "/"
