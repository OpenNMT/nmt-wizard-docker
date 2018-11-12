# Test framework

## Direct run

1\. Install dependencies:

```bash
virtualenv frameworks/test/env
source frameworks/test/env/bin/activate
pip install -r frameworks/test/requirements.txt
```

2\. Define local environment:

```bash
export WORKSPACE_DIR=/tmp/workspace
export MODELS_DIR=/tmp/models
export CORPUS_DIR=$PWD/test/corpus/
export PYTHONPATH=$PWD:$PYTHONPATH
```

3\. Run:

### Training

```bash
python frameworks/test/entrypoint.py --model_storage /tmp/saved-models \
    --config frameworks/test/config/config.json train
```

### Translation

```bash
python frameworks/test/entrypoint.py --model_storage /tmp/saved-models \
     -m MODEL_ID trans -i FILE_IN -o FILE_OUT
```
