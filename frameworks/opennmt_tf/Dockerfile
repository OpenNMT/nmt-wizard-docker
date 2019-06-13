FROM nvidia/cuda:10.0-devel-ubuntu16.04 as cuda_devel

FROM tensorflow/serving:1.13.0-gpu

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

ADD requirements.txt /root/base_requirements.txt
ADD frameworks/opennmt_tf/requirements.txt /root
RUN pip --no-cache-dir install --upgrade pip
RUN pip --no-cache-dir install -r /root/base_requirements.txt
RUN pip --no-cache-dir install -r /root/requirements.txt

ADD frameworks/opennmt_tf/entrypoint.py /root
ADD nmtwizard /root/nmtwizard

ENTRYPOINT ["python", "entrypoint.py"]
