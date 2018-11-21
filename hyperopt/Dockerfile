FROM continuumio/miniconda

RUN mkdir -p /hyperopt
WORKDIR /hyperopt

RUN conda install -c jaikumarm hyperopt
RUN conda install -c anaconda flask 

CMD ["python", "app.py"]