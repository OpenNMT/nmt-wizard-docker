# Nematus framework

## Direct run

1\. Install dependencies:

```bash
virtualenv frameworks/nematus/env
source frameworks/nematus/env/bin/activate
pip install -r frameworks/nematus/requirements.txt
```

2\. Define local environment:

```bash
export NEMATUS_DIR=$HOME/dev/nematus
export WORKSPACE_DIR=/tmp/workspace
export MODELS_DIR=/tmp/models
export CORPUS_DIR=$PWD/test/corpus/
export PYTHONPATH=$PWD:$PYTHONPATH
```

3\. Run:

### Training

```bash
python frameworks/nematus/entrypoint.py -ms /tmp/saved-models -c frameworks/nematus/config/train_ende_example.json train
```
