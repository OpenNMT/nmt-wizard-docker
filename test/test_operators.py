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

# Helper function to check filter operators.
def _is_filtered(config, tu_list):
    tu_list = _run_pipeline(config, prepoperator.ProcessType.TRAINING, tu_list)
    return len(tu_list) == 0


@pytest.mark.parametrize("config,training,text,expected", [
    (dict(), True, "hello world.", "hello world."),
    (dict(drop_word_prob=1), True, "hello world.", ""),
    (dict(drop_word_prob=1), False, "hello world.", "hello world."),
    (dict(drop_space_prob=1), True, "hello world.", "helloworld."),
    (dict(drop_char_prob=1), True, "hello world.", ""),
    (dict(drop_char_prob=1), True, "a｟a｠.", "｟a｠"),
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


def test_identity_filter():
    config = [
        {
            "op": "identity_filter",
            "min_characters": 0,
        },
    ]

    assert _is_filtered(config, tu.TranslationUnit("Hello world!", "Hello world!"))
    assert not _is_filtered(config, tu.TranslationUnit("Hello world!", "Hello world"))
    config[0]["min_characters"] = 20
    assert not _is_filtered(config, tu.TranslationUnit("Hello world!", "Hello world!"))


@pytest.mark.parametrize("filter_config,filtered", [
    (dict(), False),
    (dict(source=dict(max_characters=5)), True),
    (dict(target=dict(max_characters=5)), True),
    (dict(source=dict(max_characters=20)), False),
    (dict(source=dict(max_characters=20), target=dict(max_characters=15)), True),
    (dict(target=dict(max_characters=20, min_words=5)), True),
    (dict(source=dict(max_words=2)), True),
    (dict(min_words_ratio=1), True),
    (dict(max_words_ratio=0.5), True),
])
def test_length_filter(filter_config, filtered):
    filter_config["op"] = "length_filter"
    config = [
        {
            "op": "tokenization",
            "source": {"mode": "conservative", "joiner_annotate": True},
            "target": {"mode": "conservative", "joiner_annotate": True},
        },
        filter_config,
    ]

    source = "Hello world!"
    target = "Bonjour le monde !"
    assert filtered == _is_filtered(config, tu.TranslationUnit(source, target))
