FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

WORKDIR /root

ENV PYTHONDONTWRITEBYTECODE=1
ENV LANG=C.UTF-8

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD requirements.txt /root/base_requirements.txt
ADD frameworks/opennmt_py/requirements.txt /root
RUN python3 -m pip --no-cache-dir install -r /root/base_requirements.txt -r /root/requirements.txt

ADD frameworks/opennmt_py/entrypoint.py /root
ADD nmtwizard /root/nmtwizard

ENTRYPOINT ["python3", "entrypoint.py"]
