FROM ubuntu:16.04 as opennmt_builder

RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        wget \
        gcc \
        git \
        libboost-python-dev \
        python-dev

ARG OPENNMT_TOKENIZER_REF
ENV OPENNMT_TOKENIZER_REF=${OPENNMT_TOKENIZER_REF:-master}

RUN git clone --depth 1 --branch ${OPENNMT_TOKENIZER_REF} --single-branch https://github.com/OpenNMT/Tokenizer.git /root/tokenizer-git

WORKDIR /root/tokenizer-git

RUN mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DLIB_ONLY=ON -DWITH_PYTHON_BINDINGS=ON -DCMAKE_INSTALL_PREFIX=/root/tokenizer .. && \
    make && \
    make install && \
    cd .. && \
    rm -r build

FROM opennmt/opennmt:latest

WORKDIR /root

RUN apt-get update && apt-get install -y --no-install-recommends \
        libboost-python-dev \
        python-pip \
        python-setuptools && \
    rm -rf /var/lib/apt/lists/*

ADD frameworks/opennmt_lua/requirements.txt /root
RUN pip --no-cache-dir install -r /root/requirements.txt

ENV PYTHONPATH=/root/tokenizer/lib:$PYTHONPATH
ENV LD_LIBRARY_PATH=/root/tokenizer/lib:$LD_LIBRARY_PATH

COPY --from=opennmt_builder /root/tokenizer /root/tokenizer

ENV CORPUS_DIR=/root/corpus
ENV MODELS_DIR=/root/models

ADD frameworks/opennmt_lua/entrypoint.py /root
ADD nmtwizard /root/nmtwizard

ENTRYPOINT ["python", "entrypoint.py"]
