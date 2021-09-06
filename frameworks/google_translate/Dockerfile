FROM ubuntu:20.04

WORKDIR /root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        python3-distutils \
        wget \
        && \
    wget -nv https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    apt-get autoremove -y wget && \
    rm -rf /var/lib/apt/lists/*

ADD requirements.txt /root/base_requirements.txt
ADD frameworks/google_translate/requirements.txt /root
RUN pip --no-cache-dir install -r /root/base_requirements.txt
RUN pip --no-cache-dir install -r /root/requirements.txt

ENV PYTHONWARNINGS="ignore"

ADD frameworks/google_translate/entrypoint.py /root
ADD nmtwizard /root/nmtwizard

ENTRYPOINT ["python3", "entrypoint.py"]
