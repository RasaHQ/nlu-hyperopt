FROM rasa/rasa_nlu:0.13.8-tensorflow

RUN mkdir -p /hyperopt
WORKDIR /hyperopt

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["python", "app.py"]