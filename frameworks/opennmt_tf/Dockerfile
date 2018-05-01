FROM tensorflow/tensorflow:1.4.1-gpu

WORKDIR /root

RUN apt-get update && apt-get install -y --no-install-recommends \
        python-pip && \
    rm -rf /var/lib/apt/lists/*

ADD frameworks/opennmt_tf/requirements.txt /root
RUN pip --no-cache-dir install -r /root/requirements.txt

ENV CORPUS_DIR=/root/corpus
ENV MODELS_DIR=/root/models

ADD frameworks/opennmt_tf/entrypoint.py /root
ADD nmtwizard /root/nmtwizard

ENTRYPOINT ["python", "entrypoint.py"]
