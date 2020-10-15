# -*- coding: utf-8 -*-

import pytest
import pyonmttok

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

class _MockAligner:
    def __init__(self, alignments=None, forward_log_prob=0, backward_log_prob=0):
        self._result = {
            "forward_log_prob": forward_log_prob,
            "backward_log_prob": backward_log_prob,
            "alignments": alignments or [],
        }

    def align(self, *args):
        return self._result


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


@pytest.mark.parametrize("lower,upper,src_length,tgt_length,fwd_log_prob,bwd_log_prob,filtered", [
    (None, None, 5, 7, -5.1, -8.2, False),
    (2, None, 5, 5, 0, 0, True),  # ppl = 1
    (0, 10, 5, 5, 0, 0, False),  # ppl = 1
    (None, 0.5, 5, 5, 0, 0, True),  # ppl = 1
])
def test_align_perplexity_hard_threshold(lower,
                                         upper,
                                         src_length,
                                         tgt_length,
                                         fwd_log_prob,
                                         bwd_log_prob,
                                         filtered):
    config = [
        {
            "op": "align_perplexity_filter",
            "hard_threshold": {
                "lower": lower,
                "upper": upper,
            }
        }
    ]

    tokenizer = pyonmttok.Tokenizer("conservative", joiner_annotate=True)
    single_tu = tu.TranslationUnit(
        " ".join(str(i) for i in range(src_length)),
        " ".join(str(i) for i in range(tgt_length)),
        source_tokenizer=tokenizer,
        target_tokenizer=tokenizer)
    single_tu.set_aligner(_MockAligner(
        forward_log_prob=fwd_log_prob,
        backward_log_prob=bwd_log_prob))
    assert filtered == _is_filtered(config, single_tu)


@pytest.mark.parametrize("lower,upper,log_probs,expected_log_probs", [
    (0, 0,
     [-1.2, -3.5, -0.5, -7.0, -5.0, -1.3, -2.2, -5.8, -6.7, -18.1],
     None),
    (0.1, 0,
     [-1.2, -3.5, -0.5, -7.0, -5.0, -1.3, -5.8, -6.7, -18.1],
     None),
    (0.2, 0.1,
     [-1.2, -3.5, -0.5, -7.0, -5.0, -1.3, -2.2, -5.8, -6.7, -18.1],
     [-1.2, -3.5, -5.0, -1.3, -2.2, -5.8, -6.7]),
])
def test_align_perplexity_percent_threshold(lower, upper, log_probs, expected_log_probs):
    if expected_log_probs is None:
        expected_log_probs = log_probs
    tu_list = []
    tokenizer = pyonmttok.Tokenizer("conservative", joiner_annotate=True)
    for log_prob in log_probs:
        single_tu = tu.TranslationUnit(
            "a b c",
            "a b c",
            source_tokenizer=tokenizer,
            target_tokenizer=tokenizer)
        single_tu.set_aligner(_MockAligner(forward_log_prob=log_prob, backward_log_prob=log_prob))
        tu_list.append(single_tu)

    config = {
        "source": "en",
        "target": "fr",
        "preprocess": [
            {
                "op": "align_perplexity_filter",
                "percent_threshold": {
                    "lower": lower,
                    "upper": upper,
                }
            }
        ]
    }

    tu_list = _run_pipeline(config, prepoperator.ProcessType.TRAINING, tu_list)
    assert len(tu_list) == len(expected_log_probs)
    for single_tu, log_prob in zip(tu_list, expected_log_probs):
        assert single_tu.alignment_log_probs[0][0] == log_prob
