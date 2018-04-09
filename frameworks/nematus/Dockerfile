FROM ubuntu:16.04 as nematus_builder

RUN apt-get update && apt-get install -y \
        git

ARG NEMATUS_URL
ENV NEMATUS_URL=${NEMATUS_URL:-https://github.com/EdinburghNLP/nematus.git}
ARG NEMATUS_REF
ENV NEMATUS_REF=${NEMATUS_REF:-master}

RUN git clone --depth 1 --branch ${NEMATUS_REF} --single-branch ${NEMATUS_URL} /root/nematus


FROM nvidia/cuda:8.0-runtime-ubuntu16.04

WORKDIR /root

RUN apt-get update && apt-get install -y --no-install-recommends \
        python-pip \
        python-setuptools && \
    rm -rf /var/lib/apt/lists/*

ADD frameworks/nematus/requirements.txt /root
RUN pip --no-cache-dir install -r /root/requirements.txt

ENV NEMATUS_DIR=/root/nematus

COPY --from=nematus_builder /root/nematus $NEMATUS_DIR

ENV CORPUS_DIR=/root/corpus
ENV MODELS_DIR=/root/models

ADD frameworks/nematus/entrypoint.py /root
ADD nmtwizard /root/nmtwizard

ENTRYPOINT ["python", "entrypoint.py"]
