FROM ufoym/deepo

RUN apt-get update -y && apt-get --assume-yes install \
    build-essential cmake unzip \
    g++ gfortran wget cpio \
    libgtest-dev swig3.0 python3-dev python3-numpy 

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


# nmt-wizard-docker

ADD nmtwizard /nmtwizard
ADD utilities/tuminer/requirements.txt /
RUN pip --no-cache-dir install -r /requirements.txt && rm /requirements.txt

ADD utilities/tuminer/entrypoint.py /

ENTRYPOINT ["python3", "/entrypoint.py"]
 
