FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
# python3.7 pip 20.0.2
ENV LANG C.UTF-8

#####################################################################################################
# Install LASER
#####################################################################################################
RUN apt-get update && \
    apt-get install --no-install-recommends -y unzip \
      g++ wget cpio \
      libgtest-dev swig3.0 \
      libopenblas-dev \
      git \
      libssl-dev \
      wget && \
    git clone https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
    make -j 8 install

# python modules
RUN pip install transliterate jieba

#install faiss
RUN cd /opt && \
    git clone https://github.com/facebookresearch/faiss.git && \
    cd faiss && \
    cmake -B build . && \
    make -C build -j 8 && \
    cd build/faiss/python && python setup.py install

#install LASER
RUN cd /opt && \
    git clone https://github.com/facebookresearch/LASER.git && \
    cd LASER && \
    LASER=/opt/LASER bash ./install_models.sh

#install LASER tools-external
RUN cd /opt/LASER && \
    sed -i "s#cd fastBPE#cd fastBPE; ln -s main.cc fastBPE/fastBPE.cpp#g" install_external_tools.sh && \
    sed -i "s#python setup.py install##g" install_external_tools.sh && \
    LASER=/opt/LASER bash ./install_external_tools.sh

#install fastBPE
RUN pip install fastBPE

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
