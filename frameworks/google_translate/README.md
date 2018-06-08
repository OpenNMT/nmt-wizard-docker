# Google Translate framework

This a translation only framework that requires a [Google Translate API key](https://cloud.google.com/translate/docs/quickstart). The rest of this file assumes that the API credentials are located in `$HOME/credentials/Gateway-Translate-API.json`.

## Direct run

1\. Install dependencies:

```bash
virtualenv frameworks/google_translate/env
source frameworks/google_translate/env/bin/activate
pip install -r frameworks/google_translate/requirements.txt
```

2\. Define local environment:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=$HOME/credentials/Gateway-Translate-API.json
export WORKSPACE_DIR=/tmp/workspace
export PYTHONPATH=$PWD
```

3\. Run:

### Translation

```bash
echo "Hello world!" > /tmp/test.txt
python frameworks/google_translate/entrypoint.py \
    -c frameworks/google_translate/config/trans_ende_example.json \
    trans -i /tmp/test.txt -o /tmp/test.txt.out
cat /tmp/test.txt.out
```

## Docker run

```bash
mkdir /tmp/google_translate
echo "Hello world!" > /tmp/google_translate/test.txt

cat frameworks/google_translate/config/trans_ende_example.json | docker run -i --rm \
    -v /tmp/google_translate:/root/mount
    -v $HOME/credentials/Gateway-Translate-API.json:/root/Gateway-Translate-API.json \
    -e GOOGLE_APPLICATION_CREDENTIALS=/root/Gateway-Translate-API.json \
    nmtwizard/google-translate \
    -c - trans -i /root/mount/test.txt -o /root/mount/test.txt.out

cat /tmp/google_translate/test.txt.out
```

Credentials can also be directly passed by value:
```bash
cat frameworks/google_translate/config/trans_ende_example.json | docker run -i --rm \
    -v /tmp/google_translate:/root/mount
    -e GOOGLE_APPLICATION_CREDENTIALS="{...}" \
    nmtwizard/google-translate \
    -c - trans -i /root/mount/test.txt -o /root/mount/test.txt.out
```

