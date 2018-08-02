# Baidu Translate framework

This a translation only framework from [Baidu](https://fanyi-api.baidu.com/api/trans/product/index).

## Direct run

1\. Install dependencies:

```bash
virtualenv frameworks/baidu_translate/env
source frameworks/baidu_translate/env/bin/activate
pip install -r frameworks/baidu_translate/requirements.txt
```

2\. Define local environment:

```bash
export BAIDU_APPID=<<MY_APP_ID>>
export BAIDU_KEY=<<MY_KEY>>
export WORKSPACE_DIR=/tmp/workspace
export MODELS_DIR=/tmp/models
export PYTHONPATH=$PWD
```

3\. Run:

### Translation

```bash
echo 'Hello world!' > /tmp/test.txt
python frameworks/baidu_translate/entrypoint.py \
    -c frameworks/baidu_translate/config/trans_ende_example.json \
    trans -i /tmp/test.txt -o /tmp/test.txt.out
cat /tmp/test.txt.out
```

## Docker run

```bash
mkdir /tmp/baidu_translate
echo 'Hello world!' > /tmp/baidu_translate/test.txt

cat frameworks/baidu_translate/config/trans_ende_example.json | docker run -i --rm \
    -v /tmp/baidu_translate:/root/mount \
    -e BAIDU_APPID=${BAIDU_APPID} \
    -e BAIDU_KEY=${BAIDU_KEY} \
    nmtwizard/baidu-translate \
    -c - trans -i /root/mount/test.txt -o /root/mount/test.txt.out

cat /tmp/baidu_translate/test.txt.out
```