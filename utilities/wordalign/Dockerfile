FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python-pip python-setuptools cmake \
        git \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

ADD utilities/wordalign/requirements.txt /root
RUN pip --no-cache-dir install -r /root/requirements.txt

WORKDIR /root

RUN git clone https://github.com/clab/fast_align.git
WORKDIR /root/fast_align
RUN mkdir build ; cd build ; cmake .. ; make

WORKDIR /root
ENV PATH /root/fast_align/build:${PATH}

ADD nmtwizard /root/nmtwizard
ADD utilities/wordalign/entrypoint.py /root

ENTRYPOINT ["python", "entrypoint.py"]
