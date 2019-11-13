FROM ubuntu:16.04

WORKDIR /root

RUN apt-get update && apt-get install -y  --no-install-recommends \
        python-pip \
        python-setuptools && \
    rm -rf /var/lib/apt/lists/*

ADD requirements.txt /root/base_requirements.txt
ADD frameworks/google_translate/requirements.txt /root
RUN pip --no-cache-dir install --upgrade pip
RUN pip --no-cache-dir install -r /root/base_requirements.txt
RUN pip --no-cache-dir install -r /root/requirements.txt

ENV PYTHONWARNINGS="ignore"

ADD frameworks/google_translate/entrypoint.py /root
ADD nmtwizard /root/nmtwizard

ENTRYPOINT ["python", "entrypoint.py"]
