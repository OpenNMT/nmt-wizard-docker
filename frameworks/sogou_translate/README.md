# Sogou Translate framework

This a translation only framework from [Sogou](http://deepi.sogou.com/fanyi).

## Direct run

1\. Install dependencies:

```bash
virtualenv frameworks/sogou_translate/env
source frameworks/sogou_translate/env/bin/activate
pip install -r frameworks/sogou_translate/requirements.txt
```

2\. Define local environment:

```bash
export SOGOU_PID=<<MY_PID>>
export SOGOU_KEY=<<MY_KEY>>
export WORKSPACE_DIR=/tmp/workspace
export MODELS_DIR=/tmp/models
export PYTHONPATH=$PWD
```

3\. Run:

### Translation

```bash
echo 'Hello world!' > /tmp/test.txt
python frameworks/sogou_translate/entrypoint.py \
    -c frameworks/sogou_translate/config/trans_ende_example.json \
    trans -i /tmp/test.txt -o /tmp/test.txt.out
cat /tmp/test.txt.out
```

## Docker run

```bash
mkdir /tmp/sogou_translate
echo 'Hello world!' > /tmp/sogou_translate/test.txt

cat frameworks/sogou_translate/config/trans_ende_example.json | docker run -i --rm \
    -v /tmp/sogou_translate:/root/mount \
    -e SOGOU_PID=${SOGOU_PID} \
    -e SOGOU_KEY=${SOGOU_KEY} \
    nmtwizard/sogou-translate \
    -c - trans -i /root/mount/test.txt -o /root/mount/test.txt.out

cat /tmp/sogou_translate/test.txt.out
```