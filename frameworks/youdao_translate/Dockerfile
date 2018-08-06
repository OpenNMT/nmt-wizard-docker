FROM ubuntu:16.04

WORKDIR /root

RUN apt-get update && apt-get install -y  --no-install-recommends \
        python-pip \
        python-setuptools && \
    rm -rf /var/lib/apt/lists/*

ADD frameworks/youdao_translate/requirements.txt /root
RUN pip install -r /root/requirements.txt

ADD frameworks/youdao_translate/entrypoint.py /root
ADD nmtwizard /root/nmtwizard

ENTRYPOINT ["python", "entrypoint.py"]
