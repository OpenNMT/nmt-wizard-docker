FROM ubuntu:16.04

WORKDIR /root

RUN apt-get update && apt-get install -y --no-install-recommends \
        python-pip \
        python-setuptools && \
    rm -rf /var/lib/apt/lists/*

ADD requirements.txt /root
RUN pip --no-cache-dir install --upgrade pip
RUN pip --no-cache-dir install -r /root/requirements.txt

ENV CORPUS_DIR=/root/corpus
ENV MODELS_DIR=/root/models

ADD frameworks/test/entrypoint.py /root
ADD nmtwizard /root/nmtwizard

ENTRYPOINT ["python", "entrypoint.py"]
