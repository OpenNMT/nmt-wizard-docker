FROM ubuntu:16.04 as opennmt_builder

RUN apt-get update && apt-get install -y \
        git

ARG OPENNMT_TF_URL
ENV OPENNMT_TF_URL=${OPENNMT_TF_URL:-https://github.com/OpenNMT/OpenNMT-tf.git}
ARG OPENNMT_TF_REF
ENV OPENNMT_TF_REF=${OPENNMT_TF_REF:-v1.0.1}

RUN git clone --recursive --depth 1 --branch ${OPENNMT_TF_REF} --single-branch ${OPENNMT_TF_URL} /root/opennmt-tf


FROM tensorflow/tensorflow:1.4.1-gpu

WORKDIR /root

RUN apt-get update && apt-get install -y --no-install-recommends \
        python-pip && \
    rm -rf /var/lib/apt/lists/*

ADD frameworks/opennmt_tf/requirements.txt /root
RUN pip --no-cache-dir install -r /root/requirements.txt

ENV OPENNMT_TF_DIR=/root/opennmt-tf

COPY --from=opennmt_builder /root/opennmt-tf $OPENNMT_TF_DIR

ENV CORPUS_DIR=/root/corpus
ENV MODELS_DIR=/root/models

ADD frameworks/opennmt_tf/entrypoint.py /root
ADD nmtwizard /root/nmtwizard

ENTRYPOINT ["python", "entrypoint.py"]
