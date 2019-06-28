FROM ubuntu:16.04

WORKDIR /root

RUN apt-get update && apt-get install -y --no-install-recommends \
        python-pip \
        git \
        perl \
        default-jre \
        libsort-naturally-perl \
        libxml-parser-perl \
        libxml-twig-perl \
        wget \
        python-setuptools && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /root/tools

ARG OTEMUTEM_URL
ENV OTEMUTEM_URL=${OTEMUTEM_URL:-https://github.com/DeepLearnXMU/Otem-Utem.git}
ARG OTEMUTEM_REF
ENV OTEMUTEM_REF=${OTEMUTEM_REF:-8b4891827d6e4894ebb364284eb38b4cce57cb5e}

RUN git clone --depth 1 --single-branch ${OTEMUTEM_URL} /root/OTEMUTEM && \
	cd OTEMUTEM && git checkout ${OTEMUTEM_REF} && cd / && \
	mkdir /root/tools/Otem-Utem && \
	cp /root/OTEMUTEM/*.py /root/tools/Otem-Utem && \
	rm -rf /root/OTEMUTEM

ARG METEOR_URL
ENV METEOR_URL=${METEOR_URL:-http://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz}

RUN wget $METEOR_URL && \
  tar xvf meteor-1.5.tar.gz && \
  mv meteor-1.5 /root/tools/METEOR && \
  rm meteor-1.5.tar.gz

ADD utilities/score/BLEU /root/tools/BLEU
ADD utilities/score/TER /root/tools/TER
ADD utilities/score/NIST /root/tools/NIST

ADD requirements.txt /root
RUN pip --no-cache-dir install --upgrade pip
RUN pip --no-cache-dir install -r /root/requirements.txt

ADD nmtwizard /root/nmtwizard
ADD utilities/score/entrypoint.py /root/

ENTRYPOINT ["python", "entrypoint.py"]
