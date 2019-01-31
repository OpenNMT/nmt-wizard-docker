FROM ubuntu:16.04

WORKDIR /root

RUN apt-get update && apt-get install -y --no-install-recommends \
        python-pip \
        git \
        perl \
        python-setuptools && \
    rm -rf /var/lib/apt/lists/*

ADD utilities/score/requirements.txt /root
RUN pip --no-cache-dir install -r /root/requirements.txt

ADD nmtwizard /root/nmtwizard

ADD utilities/score/entrypoint.py /root/
ADD utilities/score/BLEU /root/tools/BLEU
ADD utilities/score/TER /root/tools/TER

ARG OTEMUTEM_URL
ENV OTEMUTEM_URL=${OTEMUTEM_URL:-https://github.com/DeepLearnXMU/Otem-Utem.git}
ARG OTEMUTEM_REF
ENV OTEMUTEM_REF=${OTEMUTEM_REF:-master}

RUN git clone --depth 1 --single-branch ${OTEMUTEM_URL} /root/OTEMUTEM && \
	cd OTEMUTEM && git checkout ${OTEMUTEM_REF} && cd / && \
	mkdir /root/tools/Otem-Utem && \
	cp /root/OTEMUTEM/*.py /root/tools/Otem-Utem && \
	rm -rf /root/OTEMUTEM

ENTRYPOINT ["python", "entrypoint.py"]
