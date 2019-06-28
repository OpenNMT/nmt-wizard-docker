FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

ARG OPENNMT_PY_VERSION
ENV OPENNMT_PY_VERSION=${OPENNMT_PY_VERSION:-92a63eebff50376579aa6bf9d09232299c67c5aa}
ENV OPENNMT_PY_DIR=/root/opennmt-py
RUN curl -fSsL -O https://github.com/OpenNMT/OpenNMT-py/archive/${OPENNMT_PY_VERSION}.tar.gz && \
    tar xf *.tar.gz && \
    mv OpenNMT-py-* ${OPENNMT_PY_DIR} && \
    rm *tar.gz

WORKDIR ${OPENNMT_PY_DIR}
RUN python setup.py install
WORKDIR /root

ADD requirements.txt /root/base_requirements.txt
ADD frameworks/opennmt_py/requirements.txt /root
RUN pip --no-cache-dir install --upgrade pip
RUN pip --no-cache-dir install -r /root/base_requirements.txt
RUN pip --no-cache-dir install -r /root/requirements.txt
RUN pip --no-cache-dir install -r ${OPENNMT_PY_DIR}/requirements.txt

ADD frameworks/opennmt_py/entrypoint.py /root
ADD nmtwizard /root/nmtwizard

ENTRYPOINT ["python", "entrypoint.py"]
