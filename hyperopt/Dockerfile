FROM rasa/rasa_nlu:0.14.0a1-tensorflow

RUN mkdir -p /hyperopt
WORKDIR /hyperopt

COPY requirements.txt .
RUN pip install -r requirements.txt

ENTRYPOINT []
CMD ["python", "app.py"]