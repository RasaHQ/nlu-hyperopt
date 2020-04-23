FROM rasa/rasa:1.9.6-full

COPY setup.py .
COPY requirements.txt .
COPY data /data
COPY nlu_hyperopt /nlu_hyperopt
COPY scripts /scripts

# Be root
USER root

RUN pip install -U pip && pip install -r requirements.txt && chmod -R +x /scripts 

RUN pip install .

WORKDIR "/"

ENTRYPOINT ["python"]

CMD ["-m", "nlu_hyperopt.app"]
