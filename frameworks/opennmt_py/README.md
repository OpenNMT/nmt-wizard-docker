# OpenNMT-py framework

## Direct run

1\. Install dependencies:

```bash
virtualenv frameworks/opennmt_py/env
source frameworks/opennmt_py/env/bin/activate
pip install -r frameworks/opennmt_py/requirements.txt
```

2\. Define local environment:

```bash
export OPENNMT_PY_DIR=$HOME/dev/OpenNMT-py
export MODELS_DIR=/tmp/models
export WORKSPACE_DIR=/tmp/workspace
export CORPUS_DIR=$PWD/test/corpus/
export PYTHONPATH=$PWD:$PYTHONPATH
```

3\. Run:

### Training

```bash
python frameworks/opennmt_py/entrypoint.py --model_storage /tmp/saved-models \
    --config frameworks/opennmt_py/config/train_ende_example.json train
```
