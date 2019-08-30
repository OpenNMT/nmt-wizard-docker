# Similarity Utility

## Direct run

1\. Install dependencies:

```bash
virtualenv utilities/similarity/env
source utilities/similarity/env/bin/activate
pip install -r utilities/similarity/requirements.txt
```

2\. Define local environment:

```bash
export MODELS_DIR=/tmp/models
export WORKSPACE_DIR=/tmp/workspace
export CORPUS_DIR=$PWD/test/corpus/
export PYTHONPATH=$PWD:$PYTHONPATH
```

3\. Run:

### Training

```bash
python utilities/similarity/entrypoint.py apply -mdir MODELPATH -tst ${CORPUS_DIR}/train/europarl-v7.de-en.10K.tok.de,${CORPUS_DIR}/train/europarl-v7.de-en.10K.tok.en -output -
```
