FROM ubuntu:20.04

WORKDIR /root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD requirements.txt /root/base_requirements.txt
ADD frameworks/google_translate/requirements.txt /root
RUN python3 -m pip --no-cache-dir install -r /root/requirements.txt -r /root/base_requirements.txt

ENV PYTHONWARNINGS="ignore"

ADD frameworks/google_translate/entrypoint.py /root
ADD nmtwizard /root/nmtwizard

ENTRYPOINT ["python3", "entrypoint.py"]
