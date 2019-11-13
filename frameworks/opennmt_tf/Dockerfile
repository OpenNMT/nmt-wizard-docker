FROM tensorflow/serving:1.14.0-gpu

WORKDIR /root
ADD requirements.txt /root/base_requirements.txt
ADD frameworks/opennmt_tf/requirements.txt /root
ADD pip_install.sh .
RUN sh pip_install.sh -r base_requirements.txt -r requirements.txt

ADD frameworks/opennmt_tf/entrypoint.py /root
ADD nmtwizard /root/nmtwizard

ENTRYPOINT ["python", "entrypoint.py"]
