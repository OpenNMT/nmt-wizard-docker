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
python utilities/score/entrypoint.py score -i test/corpus/eval/tgt.txt -r test/corpus/eval/ref.txt -l en
```

### Docker run

```bash
docker run -i \
  -v $PWD/test/corpus:/root/corpus \
  utilities/score \
  score \
  -i /root/corpus/eval/tgt.txt \
  -r /root/corpus/eval/ref.txt \
  -l en
```
