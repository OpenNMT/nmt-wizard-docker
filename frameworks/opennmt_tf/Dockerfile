FROM nvidia/cuda:9.0-devel-ubuntu16.04 as cuda_devel

FROM tensorflow/serving:1.12.0-gpu

COPY --from=cuda_devel \
     /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/

# Fix to support running without nvidia-docker.
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

WORKDIR /root

RUN apt-get update && apt-get install -y --no-install-recommends \
        python-pip \
        python-setuptools && \
    rm -rf /var/lib/apt/lists/*

ADD frameworks/opennmt_tf/requirements.txt /root
RUN pip --no-cache-dir install -r /root/requirements.txt && \
    pip --no-cache-dir install 'grpcio>=1.0,<2' && \
    pip --no-cache-dir install --no-deps 'tensorflow-serving-api==1.12.0'

ENV CORPUS_DIR=/root/corpus
ENV MODELS_DIR=/root/models

ADD frameworks/opennmt_tf/entrypoint.py /root
ADD nmtwizard /root/nmtwizard

ENTRYPOINT ["python", "entrypoint.py"]
