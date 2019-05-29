#####################################################################################################
# Pytorch image from: https://github.com/ufoym/deepo/blob/master/docker/Dockerfile.pytorch-py36-cu100
#####################################################################################################
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ENV LANG C.UTF-8
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \

    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \

    apt-get update && \

# tools
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        && \

    $GIT_CLONE https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
    make -j"$(nproc)" install && \

# python
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.6 \
        python3.6-dev \
        python3-distutils-extra \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python && \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        numpy \
        scipy \
        pandas \
        cloudpickle \
        scikit-learn \
        matplotlib \
        Cython \
        && \

# pytorch
    $PIP_INSTALL \
        future \
        numpy \
        protobuf \
        enum34 \
        pyyaml \
        typing \
        torchvision_nightly \
        && \
    $PIP_INSTALL \
        torch_nightly -f \
        https://download.pytorch.org/whl/nightly/cu100/torch_nightly.html
#####################################################################################################
# Install LASER
#####################################################################################################
RUN apt-get install --no-install-recommends -y unzip \
    g++ wget cpio \
    libgtest-dev swig3.0 \
    libopenblas-dev

# python modules
RUN pip install transliterate jieba

#install faiss
RUN cd /opt && \
    git clone https://github.com/facebookresearch/faiss.git && \
    cd faiss && \
    ./configure --with-cuda=/usr/local/cuda --with-swig=swig3.0 && \
    make -j 8 && make -C python && make install && make -C python install

#install LASER
RUN cd /opt && \
    git clone https://github.com/facebookresearch/LASER.git && \
    cd LASER && \
    LASER=/opt/LASER bash ./install_models.sh

#install LASER tools-external
RUN cd /opt/LASER && sed -i "s#g++ -std=c++11 -pthread -O3 fast.cc -o fast#g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast#g" install_external_tools.sh && \
    LASER=/opt/LASER bash ./install_external_tools.sh

#install fastBPE
RUN cd /opt/LASER/tools-external/fastBPE && python setup.py install

#install mecab
RUN cd /tmp && \
    wget -O mecab-0.996.tar.gz "https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7cENtOXlicTFaRUE" && \
    tar xvzf mecab-0.996.tar.gz && \
    cd mecab-0.996 && \
    ./configure --prefix /opt/LASER/tools-external/mecab --with-charset=utf8 && \
    make install && \
    rm -rf mecab-0.996.tar.gz mecab-0.996

RUN cd /tmp && \
    wget -O mecab-ipadic-2.7.0-XXXX.tar.gz "https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7MWVlSDBCSXZMTXM" && \
    tar xvzf mecab-ipadic-2.7.0-XXXX.tar.gz && \
    cd mecab-ipadic-2.7.0-20070801/ && \
    ./configure --prefix=/opt/LASER/tools-external/mecab --with-mecab-config=/opt/LASER/tools-external/mecab/bin/mecab-config --with-charset=utf8 && \
    make install && \
    rm -rf mecab-ipadic-2.7.0-XXXX.tar.gz mecab-ipadic-2.7.0-20070801

RUN echo "export LASER=/opt/LASER" >> /etc/bash.bashrc && echo "export LC_ALL=C.UTF-8" >> /etc/bash.bashrc
#####################################################################################################
# config & cleanup
#####################################################################################################
RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*
#####################################################################################################
# nmt-wizard-docker
#####################################################################################################
ADD nmtwizard /nmtwizard
ADD requirements.txt /
RUN pip --no-cache-dir install -r /requirements.txt && rm /requirements.txt

ADD utilities/tuminer/entrypoint.py /
ENTRYPOINT ["python3", "/entrypoint.py"]
