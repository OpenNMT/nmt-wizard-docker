# OpenNMT-tf framework

## Direct run

1\. Install dependencies:

```bash
virtualenv frameworks/opennmt_tf/env
source frameworks/opennmt_tf/env/bin/activate
pip install -r frameworks/opennmt_tf/requirements.txt
pip install --upgrade tensorflow==1.4.1
```

2\. Define local environment:

```bash
export OPENNMT_TF_DIR=$HOME/dev/OpenNMT-tf
export MODELS_DIR=/tmp/models
export WORKSPACE_DIR=/tmp/workspace
export CORPUS_DIR=$PWD/test/corpus/
export PYTHONPATH=$PWD:$OPENNMT_TF_DIR:$PYTHONPATH
```

3\. Run:

### Training

```bash
python frameworks/opennmt_tf/entrypoint.py --model_storage /tmp/saved-models \
    --config frameworks/opennmt_tf/config/train_ende_example.json train
```
