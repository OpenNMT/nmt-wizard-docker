FROM tensorflow/serving:1.8.0-devel-gpu as serving_builder

FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

COPY --from=serving_builder \
     /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/
COPY --from=serving_builder \
     /usr/local/bin/tensorflow_model_server /usr/local/bin/

# Fix to support running without nvidia-docker.
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

WORKDIR /root

ENV NCCL_VERSION=2.2.12-1+cuda9.0
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        libnccl2=${NCCL_VERSION} \
        libgomp1 \
        python-pip \
        python-setuptools && \
    rm -rf /var/lib/apt/lists/*

ADD frameworks/opennmt_tf/requirements.txt /root
RUN pip --no-cache-dir install -r /root/requirements.txt && \
    pip --no-cache-dir install 'grpcio>=1.0' && \
    pip --no-cache-dir install --no-deps 'tensorflow-serving-api==1.8.0'

ENV CORPUS_DIR=/root/corpus
ENV MODELS_DIR=/root/models

ADD frameworks/opennmt_tf/entrypoint.py /root
ADD nmtwizard /root/nmtwizard

ENTRYPOINT ["python", "entrypoint.py"]
