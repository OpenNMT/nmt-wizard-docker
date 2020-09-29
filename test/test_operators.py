# -*- coding: utf-8 -*-

import pytest

from nmtwizard.preprocess import prepoperator
from nmtwizard.preprocess import loader
from nmtwizard.preprocess import tu


# Helper function to run the pipeline on a string or TU.
def _run_pipeline(config, process_type, tu_list):
    if isinstance(tu_list, str):
        tu_list = tu.TranslationUnit(tu_list)
    if not isinstance(tu_list, list):
        tu_list = [tu_list]
    if isinstance(config, list):
        config = {
            "source": "xx",
            "target": "yy",
            "preprocess": config,
        }
    pipeline = prepoperator.Pipeline(config, process_type)
    tu_list, _ = pipeline((tu_list, {}))
    return tu_list


@pytest.mark.parametrize("config,training,text,expected", [
    (dict(), True, "hello world.", "hello world."),
    (dict(drop_word_prob=1), True, "hello world.", ""),
    (dict(drop_word_prob=1), False, "hello world.", "hello world."),
    (dict(drop_space_prob=1), True, "hello world.", "helloworld."),
    (dict(drop_char_prob=1), True, "hello world.", ""),
    (dict(duplicate_char_prob=1), True, "hello.", "hheelllloo.."),
    (dict(swap_char_prob=1), True, "hello.", "ehllo."),
])
def test_noise(config, training, text, expected):
    config_base = [
        {
            "op": "tokenization",
            "source": {"mode": "conservative", "joiner_annotate": True},
            "target": {"mode": "conservative", "joiner_annotate": True},
        },
        {
            "op": "noise",
        },
    ]

    config_base[-1].update(config)
    process_type = (prepoperator.ProcessType.TRAINING if training
                    else prepoperator.ProcessType.INFERENCE)
    tu_list = _run_pipeline(config_base, process_type, text)
    assert tu_list[0].src_detok == expected
