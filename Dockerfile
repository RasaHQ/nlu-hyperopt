FROM rasa/rasa:3.2.4-full

COPY setup.py .
COPY requirements.txt .
COPY train_test_split /train_test_split
COPY nlu_hyperopt /nlu_hyperopt
COPY scripts /scripts

# Be root
USER root

RUN pip install -U pip && pip install -r requirements.txt && chmod -R +x /scripts  

ENV PYTHONPATH "/"

ENTRYPOINT ["python"]

CMD ["-m", "nlu_hyperopt.app"]
