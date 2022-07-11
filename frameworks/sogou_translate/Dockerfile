FROM ubuntu:20.04

WORKDIR /root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD requirements.txt /root
RUN python3 -m pip --no-cache-dir install -r /root/requirements.txt

ADD frameworks/sogou_translate/entrypoint.py /root
ADD nmtwizard /root/nmtwizard

ENTRYPOINT ["python3", "entrypoint.py"]
