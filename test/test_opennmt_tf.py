import itertools
import json
import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
framework_path = os.path.join(test_dir, "..", "frameworks", "opennmt_tf")
sys.path.insert(0, framework_path)

import tensorflow as tf

from entrypoint import OpenNMTTFFramework
from test_framework import _run_framework


config_base = {
    "source": "en",
    "target": "de",
    "data": {
        "sample": 4,
        "sample_dist": [
            {
                "path": os.path.join(test_dir, "corpus", "train"),
                "distribution": [["europarl", 1]],
            }
        ],
    },
    "preprocess": [
        {
            "op": "tokenization",
            "source": {"mode": "space"},
            "target": {"mode": "space"},
        }
    ],
    "vocabulary": {
        "source": {"path": "${CORPUS_DIR}/vocab/en-vocab.txt"},
        "target": {"path": "${CORPUS_DIR}/vocab/de-vocab.txt"},
    },
    "options": {
        "model_type": "TransformerTiny",
        "auto_config": True,
        "config": {
            "train": {
                "batch_size": 2,
                "batch_type": "examples",
                "effective_batch_size": None,
                "length_bucket_width": None,
            },
        },
    },
}


def test_train(tmpdir):
    sample_size = config_base["data"]["sample"]
    batch_size = config_base["options"]["config"]["train"]["batch_size"]

    for iteration in range(2):
        model_dir, result = _run_framework(
            tmpdir,
            "model%d" % iteration,
            "train",
            parent="model%d" % (iteration - 1) if iteration > 0 else None,
            config=config_base,
            framework_fn=OpenNMTTFFramework,
            return_output=True,
        )

        assert result["num_sentences"] == sample_size
        assert result["num_steps"] == sample_size // batch_size
        assert result["last_step"] == result["num_steps"] * (iteration + 1)

        assert "model_description.py" not in os.listdir(model_dir)

        checkpoint_state = tf.train.get_checkpoint_state(model_dir)
        assert len(checkpoint_state.all_model_checkpoint_paths) == 1

        last_checkpoint = tf.train.latest_checkpoint(model_dir)
        assert int(last_checkpoint.split("-")[-1]) == result["last_step"]


def _copy_lines(src, dst, begin, end):
    with open(src) as src_file, open(dst, "w") as dst_file:
        for line in itertools.islice(src_file, begin, end):
            dst_file.write(line)


def test_eval(tmpdir):
    _run_framework(
        tmpdir,
        "model0",
        "train",
        config=config_base,
        framework_fn=OpenNMTTFFramework,
    )

    data_dir = os.path.join(test_dir, "corpus", "train")
    src_train = os.path.join(data_dir, "europarl-v7.de-en.10K.tok.en")
    tgt_train = os.path.join(data_dir, "europarl-v7.de-en.10K.tok.en")

    src_valids = []
    tgt_valids = []

    for i in range(3):
        begin_line = 10 * i
        end_line = 10 * (i + 1)

        src_valid = str(tmpdir.join("valid.src.%d" % i))
        tgt_valid = str(tmpdir.join("valid.tgt.%d" % i))

        _copy_lines(src_train, src_valid, begin_line, end_line)
        _copy_lines(tgt_train, tgt_valid, begin_line, end_line)

        src_valids.append(src_valid)
        tgt_valids.append(tgt_valid)

    output_path = str(tmpdir.join("output"))

    _run_framework(
        tmpdir,
        "eval",
        "eval -s %s -t %s -o %s"
        % (" ".join(src_valids), " ".join(tgt_valids), output_path),
        parent="model0",
        config=config_base,
        framework_fn=OpenNMTTFFramework,
    )

    with open(output_path) as output_file:
        result = json.load(output_file)

    assert 0 < result["all"]["loss"] < 10
    for tgt_valid in tgt_valids:
        assert 0 < result["files"][tgt_valid]["loss"] < 10
