# OpenNMT-lua framework

## Direct run

1\. Install dependencies:

```bash
virtualenv frameworks/opennmt_lua/env
source frameworks/opennmt_lua/env/bin/activate
pip install -r frameworks/opennmt_lua/requirements.txt
```

2\. Define local environment:

```bash
export ONMT_DIR=$HOME/dev/OpenNMT
export WORKSPACE_DIR=/tmp/workspace
export MODELS_DIR=/tmp/models
export CORPUS_DIR=$PWD/test/corpus/
export PYTHONPATH=$PWD:$PYTHONPATH
```

3\. Run:

### Training

```bash
python frameworks/opennmt_lua/entrypoint.py --model_storage /tmp/saved-models \
    --config frameworks/opennmt_lua/config/train_ende_example.json train
```
