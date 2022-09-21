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
