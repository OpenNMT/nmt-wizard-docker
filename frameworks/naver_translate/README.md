# Naver Translate framework

This a translation only framework from [Naver](https://developers.naver.com/docs/nmt/reference/).

## Direct run

1\. Install dependencies:

```bash
virtualenv frameworks/naver_translate/env
source frameworks/naver_translate/env/bin/activate
pip install -r frameworks/naver_translate/requirements.txt
```

2\. Define local environment:

```bash
export NAVER_CLIENT_ID=<<MY_CLIENT_ID>>
export NAVER_SECRET=<<MY_SECRET>>
export WORKSPACE_DIR=/tmp/workspace
export MODELS_DIR=/tmp/models
export PYTHONPATH=$PWD
```

3\. Run:

### Translation

```bash
echo 'Hello world!' > /tmp/test.txt
python frameworks/naver_translate/entrypoint.py \
    -c frameworks/naver_translate/config/trans_enko_example.json \
    trans -i /tmp/test.txt -o /tmp/test.txt.out
cat /tmp/test.txt.out
```

## Docker run

```bash
mkdir /tmp/naver_translate
echo 'Hello world!' > /tmp/naver_translate/test.txt

cat frameworks/naver_translate/config/trans_ende_example.json | docker run -i --rm \
    -v /tmp/naver_translate:/root/mount \
    -e NAVER_CLIENT_ID=${NAVER_CLIENT_ID} \
    -e NAVER_SECRET=${NAVER_SECRET} \
    nmtwizard/naver-translate \
    -c - trans -i /root/mount/test.txt -o /root/mount/test.txt.out

cat /tmp/naver_translate/test.txt.out
```