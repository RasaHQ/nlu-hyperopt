FROM continuumio/miniconda

RUN mkdir -p /hyperopt
WORKDIR /hyperopt

RUN conda install -c jaikumarm hyperopt
RUN conda install -c anaconda flask 

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["python", "app.py"]