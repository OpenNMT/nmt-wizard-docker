FROM ubuntu:16.04

WORKDIR /root

RUN apt-get update && apt-get install -y --no-install-recommends \
        python-pip \
        perl \
        python-setuptools && \
    rm -rf /var/lib/apt/lists/*

ADD utilities/score/requirements.txt /root
RUN pip --no-cache-dir install -r /root/requirements.txt

ADD nmtwizard /root/nmtwizard

ADD utilities/score/entrypoint.py /root/
ADD utilities/score/BLEU /root/tools/BLEU
ADD utilities/score/TER /root/tools/TER

ENTRYPOINT ["python", "entrypoint.py"]
