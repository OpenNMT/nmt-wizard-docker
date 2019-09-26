FROM tensorflow/tensorflow:1.14.0-gpu

WORKDIR /root

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake && \
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

RUN mkdir /root/tools

ARG FAST_ALIGN_URL
ENV FAST_ALIGN_URL=${FAST_ALIGN_URL:-https://github.com/clab/fast_align}
ARG FAST_ALIGN_REF
ENV FAST_ALIGN_REF=${FAST_ALIGN_REF:-master}

RUN git clone --depth 1 --single-branch ${FAST_ALIGN_URL} /root/fast_align && \
	cd fast_align && git checkout ${FAST_ALIGN_REF} && \
  mkdir build && cd build && cmake .. && make && \
	cp fast_align /root/tools && \
	rm -rf /root/fast_align

ADD nmtwizard /root/nmtwizard

ADD utilities/similarity/entrypoint.py /root/

ENTRYPOINT ["python", "entrypoint.py"]