FROM rasa/rasa_nlu:latest-tensorflow

RUN mkdir -p /hyperopt
WORKDIR /hyperopt

COPY requirements.txt .
RUN pip install -r requirements.txt

ENTRYPOINT []
CMD ["python", "app.py"]