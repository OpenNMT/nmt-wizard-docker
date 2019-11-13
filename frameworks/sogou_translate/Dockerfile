FROM ubuntu:16.04

WORKDIR /root
ADD requirements.txt /root
ADD pip_install.sh .
RUN sh pip_install.sh -r requirements.txt

ADD frameworks/sogou_translate/entrypoint.py /root
ADD nmtwizard /root/nmtwizard

ENTRYPOINT ["python", "entrypoint.py"]
