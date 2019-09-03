FROM tensorflow/tensorflow:1.14.0-gpu

WORKDIR /root

RUN apt-get update && apt-get install -y --no-install-recommends \
        git && \
    rm -rf /var/lib/apt/lists/*

ADD requirements.txt /root
RUN pip --no-cache-dir install -r /root/requirements.txt

ARG SIMILARITY_URL
ENV SIMILARITY_URL=${SIMILARITY_URL:-https://github.com/systran/similarity.git}
ARG SIMILARITY_REF
ENV SIMILARITY_REF=${SIMILARITY_REF:-master}

RUN git clone --depth 1 --single-branch ${SIMILARITY_URL} /root/similarity && \
	cd similarity && git checkout ${SIMILARITY_REF} && cd .. && \
	rm -rf similarity/.git/* && \
	cp /root/similarity/src/*.py /root/similarity/requirements.txt /root && \
	rm -rf /root/similarity

RUN pip --no-cache-dir install -r /root/requirements.txt

ADD nmtwizard /root/nmtwizard

ADD utilities/similarity/entrypoint.py /root/

ENTRYPOINT ["python", "entrypoint.py"]