# DeepL Translate framework

This a translation only framework that requires a [DeepL API key](https://www.deepl.com/pro.html) `<<MYCREDENTIAL>>`.

## Direct run

1\. Install dependencies:

```bash
virtualenv frameworks/deepl_translate/env
source frameworks/deepl_translate/env/bin/activate
pip install -r frameworks/deepl_translate/requirements.txt
```

2\. Define local environment:

```bash
export DEEPL_CREDENTIALS=<<MYCREDENTIAL>>
export WORKSPACE_DIR=/tmp/workspace
export MODELS_DIR=/tmp/models
export PYTHONPATH=$PWD
```

3\. Run:

### Translation

```bash
echo 'Hello world!' > /tmp/test.txt
python frameworks/deepl_translate/entrypoint.py \
    -c frameworks/deepl_translate/config/trans_ende_example.json \
    trans -i /tmp/test.txt -o /tmp/test.txt.out
cat /tmp/test.txt.out
```

## Docker run

```bash
mkdir /tmp/deepl_translate
echo 'Hello world!' > /tmp/deepl_translate/test.txt

cat frameworks/deepl_translate/config/trans_ende_example.json | docker run -i --rm \
    -v /tmp/deepl_translate:/root/mount \
    -e DEEPL_CREDENTIALS=${DEEPL_CREDENTIALS} \
    nmtwizard/deepl-translate \
    -c - trans -i /root/mount/test.txt -o /root/mount/test.txt.out

cat /tmp/deepl_translate/test.txt.out
```