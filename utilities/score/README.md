# Score Utility

## Direct run

1\. Install dependencies:

```bash
virtualenv utilities/score/env
source utilities/score/env/bin/activate
pip install -r utilities/score/requirements.txt
```

2\. Define local environment:

```bash
export WORKSPACE_DIR=/tmp/workspace
export TOOLS_DIR=$PWD/utilities/score
export PYTHONPATH=$PWD:$PYTHONPATH
```

3\. Run:

### Local run

```bash
python utilities/score/entrypoint.py score \
  -o test/corpus/eval/testset1.out \
     test/corpus/eval/testset2.out \
  -r test/corpus/eval/testset1.ref \
     test/corpus/eval/testset2.ref.1,test/corpus/eval/testset2.ref.2 \
  -l en
```

### Docker run

```bash
docker run -i \
  -v $PWD/test/corpus:/root/corpus \
  utilities/score \
  score \
  -o /root/corpus/eval/testset1.out \
     /root/corpus/eval/testset2.out \
  -r /root/corpus/eval/testset1.ref \
     /root/corpus/eval/testset2.ref.1,/root/corpus/eval/testset2.ref.2 \
  -l en
```
