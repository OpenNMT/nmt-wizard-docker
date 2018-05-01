FROM ubuntu:16.04 as opennmt_builder

RUN apt-get update && apt-get install -y \
        git

ARG OPENNMT_PY_URL
ENV OPENNMT_PY_URL=${OPENNMT_PY_URL:-https://github.com/OpenNMT/OpenNMT-py.git}
ARG OPENNMT_PY_REF
ENV OPENNMT_PY_REF=${OPENNMT_PY_REF:-master}

RUN git clone --depth 1 --branch ${OPENNMT_PY_REF} --single-branch ${OPENNMT_PY_URL} /root/opennmt-py


FROM nvidia/cuda:8.0-runtime-ubuntu16.04

WORKDIR /root

RUN apt-get update && apt-get install -y --no-install-recommends \
        python-pip \
        python-setuptools && \
    rm -rf /var/lib/apt/lists/*

ADD frameworks/opennmt_py/requirements.txt /root
RUN pip --no-cache-dir install -r /root/requirements.txt

ENV OPENNMT_PY_DIR=/root/opennmt-py

COPY --from=opennmt_builder /root/opennmt-py $OPENNMT_PY_DIR

ENV CORPUS_DIR=/root/corpus
ENV MODELS_DIR=/root/models

ADD frameworks/opennmt_py/entrypoint.py /root
ADD nmtwizard /root/nmtwizard

ENTRYPOINT ["python", "entrypoint.py"]
