FROM rasa/rasa:1.9.6-full

WORKDIR "/"

COPY setup.py .
COPY requirements.txt .
COPY data /data
COPY nlu_hyperopt /nlu_hyperopt
COPY scripts /scripts
COPY entrypoint.sh /entrypoint.sh

# Be root
USER root

RUN pip install -U pip && pip install -r requirements.txt && chmod -R +x /scripts  

RUN chown root /entrypoint.sh && chmod +x /entrypoint.sh

ENV PYTHONPATH "/"

ENTRYPOINT ["/entrypoint.sh"]

CMD ["-m", "nlu_hyperopt.app"]
