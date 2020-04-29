FROM rasa/rasa:1.9.6-full

COPY setup.py .
COPY requirements.txt .
COPY data .
COPY nlu_hyperopt .
COPY scripts .

# Be root
USER root

RUN pip install -U pip && pip install -r /requirements.txt && chmod -R +x /scripts  

ENV PYTHONPATH "/"

ENTRYPOINT ["python"]

CMD ["-m", "nlu_hyperopt.app"]
