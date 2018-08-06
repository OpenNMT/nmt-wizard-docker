# Youdao Translate framework

This a translation only framework from [Youdao](https://ai.youdao.com/gw.s).
Be careful, it costs money for every request.

## Direct run

1\. Install dependencies:

```bash
virtualenv frameworks/youdao_translate/env
source frameworks/youdao_translate/env/bin/activate
pip install -r frameworks/youdao_translate/requirements.txt
```

2\. Define local environment:

```bash
export YOUDAO_APPID=<<MY_APP_ID>>
export YOUDAO_KEY=<<MY_KEY>>
export WORKSPACE_DIR=/tmp/workspace
export MODELS_DIR=/tmp/models
export PYTHONPATH=$PWD
```

3\. Run:

### Translation

```bash
echo 'Hello world!' > /tmp/test.txt
python frameworks/youdao_translate/entrypoint.py \
    -c frameworks/youdao_translate/config/trans_enfr_example.json \
    trans -i /tmp/test.txt -o /tmp/test.txt.out
cat /tmp/test.txt.out
```

## Docker run

```bash
mkdir /tmp/youdao_translate
echo 'Hello world!' > /tmp/youdao_translate/test.txt

cat frameworks/youdao_translate/config/trans_enfr_example.json | docker run -i --rm \
    -v /tmp/youdao_translate:/root/mount \
    -e YOUDAO_APPID=${YOUDAO_APPID} \
    -e YOUDAO_KEY=${YOUDAO_KEY} \
    nmtwizard/youdao-translate \
    -c - trans -i /root/mount/test.txt -o /root/mount/test.txt.out

cat /tmp/youdao_translate/test.txt.out
```