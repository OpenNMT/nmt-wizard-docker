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

### Local run

If you run this utility locally, you need [similarity project](https://github.com/SYSTRAN/similarity):
```bash
git clone https://github.com/SYSTRAN/similarity utilities/similarity/similarity
pip install -r utilities/similarity/similarity/requirements.txt
```

```bash
python utilities/similarity/entrypoint.py \
  simapply \
  -mdir ${MODELS_DIR} \
  -tst_src ${CORPUS_DIR}/test.en \
  -tst_tgt ${CORPUS_DIR}/test.fr \
  -output $PWD/output
```

### Docker run

```bash
docker run -i --rm \
  -v ${MODELS_DIR}:/modelpath \
  -v ${CORPUS_DIR}:/data \
  nmtwizard/similarity:latest \
  simapply \
  -mdir /modelpath \
  -tst_src /data/test.en \
  -tst_tgt /data/test.fr \
  -output /modelpath/output
```

