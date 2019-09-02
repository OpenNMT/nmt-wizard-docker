FROM nvidia/cuda:9.0-devel-ubuntu16.04 as cuda_devel

FROM tensorflow/serving:1.11.0-gpu

COPY --from=cuda_devel \
     /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/

# Fix to support running without nvidia-docker.
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

WORKDIR /root

RUN apt-get update && apt-get install -y --no-install-recommends \
        python-pip \
        git \
        python-setuptools && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
ADD utilities/similarity/requirements.txt /root
RUN pip --no-cache-dir install -r /root/requirements.txt

ARG SIMILARITY_URL
ENV SIMILARITY_URL=${SIMILARITY_URL:-https://github.com/systran/similarity.git}
ARG SIMILARITY_REF
ENV SIMILARITY_REF=${SIMILARITY_REF:-master}

RUN git clone --depth 1 --single-branch ${SIMILARITY_URL} /root/similarity && \
	cd similarity && git checkout ${SIMILARITY_REF} && cd .. && \
	rm -rf similarity/.git/* && \
	cp /root/similarity/src/*.py /root/similarity/requirements.txt /root && \
	rm -rf /root/similarity

RUN pip --no-cache-dir install -r /root/requirements.txt

ADD nmtwizard /root/nmtwizard

ADD utilities/similarity/entrypoint.py /root/

ENTRYPOINT ["python", "entrypoint.py"]