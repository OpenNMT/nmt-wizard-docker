FROM ubuntu:16.04 as opennmt_builder

RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        gcc \
        git \
        libboost-python-dev \
        python-dev

ARG OPENNMT_PY_URL
ENV OPENNMT_PY_URL=${OPENNMT_PY_URL:-https://github.com/OpenNMT/OpenNMT-py.git}
ARG OPENNMT_PY_REF
ENV OPENNMT_PY_REF=${OPENNMT_PY_REF:-master}

RUN git clone --depth 1 --branch ${OPENNMT_PY_REF} --single-branch ${OPENNMT_PY_URL} /root/opennmt-py
RUN git clone --depth 1 --branch v1.1.1 --single-branch https://github.com/OpenNMT/Tokenizer.git /root/tokenizer-git

WORKDIR /root/tokenizer-git

RUN mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DLIB_ONLY=ON -DWITH_PYTHON_BINDINGS=ON -DCMAKE_INSTALL_PREFIX=/root/tokenizer .. && \
    make && \
    make install && \
    cd .. && \
    rm -r build


FROM nvidia/cuda:8.0-runtime-ubuntu16.04

WORKDIR /root

RUN apt-get update && apt-get install -y --no-install-recommends \
        libboost-python-dev \
        python-pip \
        python-setuptools && \
    rm -rf /var/lib/apt/lists/*

ADD frameworks/opennmt_py/requirements.txt /root
RUN pip --no-cache-dir install -r /root/requirements.txt

ENV OPENNMT_PY_DIR=/root/opennmt-py
ENV PYTHONPATH=/root/tokenizer/lib:$PYTHONPATH
ENV LD_LIBRARY_PATH=/root/tokenizer/lib:$LD_LIBRARY_PATH

COPY --from=opennmt_builder /root/opennmt-py $OPENNMT_PY_DIR
COPY --from=opennmt_builder /root/tokenizer /root/tokenizer

ENV CORPUS_DIR=/root/corpus
ENV MODELS_DIR=/root/models

ADD frameworks/opennmt_py/entrypoint.py /root
ADD nmtwizard /root/nmtwizard

ENTRYPOINT ["python", "entrypoint.py"]
