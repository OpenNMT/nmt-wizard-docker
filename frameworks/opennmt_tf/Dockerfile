FROM tensorflow/serving:1.14.0-gpu

WORKDIR /root

RUN apt-get update && apt-get install -y --no-install-recommends \
        python-pip && \
    rm -rf /var/lib/apt/lists/*

ADD requirements.txt /root/base_requirements.txt
ADD frameworks/opennmt_tf/requirements.txt /root
RUN pip --no-cache-dir install --upgrade pip setuptools
RUN pip --no-cache-dir install -r /root/base_requirements.txt
RUN pip --no-cache-dir install -r /root/requirements.txt

ADD frameworks/opennmt_tf/entrypoint.py /root
ADD nmtwizard /root/nmtwizard

ENTRYPOINT ["python", "entrypoint.py"]
