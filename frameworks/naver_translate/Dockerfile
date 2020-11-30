FROM ubuntu:18.04

WORKDIR /root

RUN apt-get update && apt-get install -y  --no-install-recommends \
        python3 \
        python3-setuptools \
        wget \
        ca-certificates \
        && \
    wget -nv https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    apt-get autoremove -y wget && \
    rm -rf /var/lib/apt/lists/*

ADD requirements.txt /root
RUN pip --no-cache-dir install --upgrade pip
RUN pip --no-cache-dir install -r /root/requirements.txt --use-feature=2020-resolver

ADD frameworks/naver_translate/entrypoint.py /root
ADD nmtwizard /root/nmtwizard

ENTRYPOINT ["python3", "entrypoint.py"]
