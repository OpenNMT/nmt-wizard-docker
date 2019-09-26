# Similarity Utility

## Direct run

1\. Install dependencies:

```bash
virtualenv utilities/similarity/env
source utilities/similarity/env/bin/activate
pip install -r requirements.txt
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
Train

```bash
python utilities/similarity/entrypoint.py \
  -g 1 \
  simtrain \
  -mdir ${MODELS_DIR} \
  -trn_src ${TESTSRC} \
  -trn_tgt ${TESTTGT} \
  -build_data_mode puid \
  -src_emb_size 64 \
  -tgt_emb_size 64 \
  -max_sents 1000000 \
  -lr_method adam -lr 0.001 \
  -n_epochs 1
```

Inference

```bash
python utilities/similarity/entrypoint.py \
  simapply \
  -mdir ${MODELS_DIR} \
  -tst_src ${CORPUS_DIR}/test.en \
  -tst_tgt ${CORPUS_DIR}/test.fr \
  -output $PWD/output
```

### Docker run

Train

```bash
nvidia-docker run -i --rm \
  -v ${MODELS_DIR}:/modelpath \
  -v ${CORPUS_DIR}:/data \
  nmtwizard/similarity:latest \
  -g 1 \
  simtrain \
  -mdir /modelpath \
  -trn_src /data/train.en \
  -trn_tgt /data/train.fr \
  -build_data_mode puid \
  -src_emb_size 64 \
  -tgt_emb_size 64 \
  -max_sents 1000000 \
  -lr_method adam -lr 0.001 \
  -n_epochs 1
```

Inference
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

