FROM ubuntu:16.04

WORKDIR /root
ADD requirements.txt /root/base_requirements.txt
ADD frameworks/google_translate/requirements.txt /root
ADD pip_install.sh .
RUN sh pip_install.sh -r base_requirements.txt -r requirements.txt

ENV PYTHONWARNINGS="ignore"

ADD frameworks/google_translate/entrypoint.py /root
ADD nmtwizard /root/nmtwizard

ENTRYPOINT ["python", "entrypoint.py"]
