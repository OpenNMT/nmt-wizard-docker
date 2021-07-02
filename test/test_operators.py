# -*- coding: utf-8 -*-

import os
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
            "source": "en",
            "target": "fr",
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


def test_tokenization_with_vocabulary_restriction(tmpdir):
    sp_model_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "corpus",
        "resources",
        "subword",
        "en_de.sp",
    )
    config = {
        "source": "en",
        "target": "de",
        "preprocess": [
            {
                "op": "tokenization",
                "source": {
                    "mode": "none",
                    "sp_model_path": sp_model_path,
                    "restrict_subword_vocabulary": True,
                },
                "target": {
                    "mode": "none",
                    "sp_model_path": sp_model_path,
                },
            },
        ],
    }

    process_type = prepoperator.ProcessType.INFERENCE
    example = tu.TranslationUnit("World", "World")

    with pytest.raises(ValueError, match="restrict_subword_vocabulary"):
        pipeline = prepoperator.Pipeline(config, process_type)

    vocab_path = str(tmpdir.join("vocab.txt"))
    with open(vocab_path, "w") as vocab_file:
        vocab_file.write("# Comment\n")
        vocab_file.write("▁Wor 0.0224656\n")
    config.update(
        {
            "vocabulary": {
                "source": {
                    "path": vocab_path,
                },
                "target": {
                    "path": vocab_path,
                },
            },
        }
    )

    pipeline = prepoperator.Pipeline(config, process_type)
    tu_list, _ = pipeline(([example], {}))

    assert tu_list[0].src_tok.tokens[0] == ["▁Wor", "l", "d"]
    assert tu_list[0].tgt_tok.tokens[0] == ["▁World"]


def test_tokenization_with_lang():
    tokenization_config = {
        "mode": "aggressive",
        "case_markup": True,
        "soft_case_regions": True,
    }
    config = {
        "source": "el",
        "target": "en",
        "preprocess": [
            {
                "op": "tokenization",
                "source": tokenization_config,
                "target": tokenization_config,
            }
        ],
    }

    example = tu.TranslationUnit("ΣΙΓΜΑ ΤΕΛΙΚΟΣ")
    pipeline = prepoperator.Pipeline(config, prepoperator.ProcessType.INFERENCE)
    tu_list, _ = pipeline(([example], {}))

    assert tu_list[0].src_tok.tokens[0] == [
        "｟mrk_begin_case_region_U｠",
        "σιγμα",
        "τελικος",
        "｟mrk_end_case_region_U｠",
    ]


def test_tokenization_with_non_iso_639_lang():
    config = {
        "source": "en-GB",
        "target": "en-US",
        "preprocess": [
            {
                "op": "tokenization",
                "source": {"mode": "none"},
                "target": {"mode": "none"},
            }
        ],
    }

    # Should not throw an exception.
    prepoperator.Pipeline(config, prepoperator.ProcessType.INFERENCE)


def test_tokenization_with_inference_config(tmpdir):
    config = {
        "source": "en",
        "target": "de",
        "preprocess": [
            {
                "op": "tokenization",
                "source": {
                    "mode": "aggressive",
                },
                "target": {
                    "mode": "aggressive",
                },
            },
        ],
    }

    process_type = prepoperator.ProcessType.INFERENCE
    example = tu.TranslationUnit("2,000", "2,000")

    pipeline = prepoperator.Pipeline(config, process_type)

    tu_list, _ = pipeline(([example], {}))

    assert tu_list[0].src_tok.tokens[0] == ["2", ",", "000"]
    assert tu_list[0].tgt_tok.tokens[0] == ["2", ",", "000"]

    config["inference"] = {
        "overrides": {"tokenization_1": {"source": {"mode": "none"}}}
    }
    pipeline = prepoperator.Pipeline(config, process_type)

    example = tu.TranslationUnit("2,000", "2,000")
    tu_list, _ = pipeline(([example], {}))

    assert tu_list[0].src_tok.tokens[0] == ["2,000"]
    assert tu_list[0].tgt_tok.tokens[0] == ["2", ",", "000"]


@pytest.mark.parametrize(
    "config,training,text,expected",
    [
        (dict(), True, "hello world.", "hello world."),
        (dict(drop_word_prob=1), True, "hello world.", ""),
        (dict({"source": {"drop_word_prob": 1}}), True, "hello world.", ""),
        (dict(drop_word_prob=1), False, "hello world.", "hello world."),
        (dict(drop_space_prob=1), True, "hello world.", "helloworld."),
        (dict(drop_char_prob=1), True, "hello world.", ""),
        (dict(drop_char_prob=1), True, "a｟a｠.", "｟a｠"),
        (dict(duplicate_char_prob=1), True, "hello.", "hheelllloo.."),
        (dict(swap_char_prob=1), True, "hello.", "ehllo."),
    ],
)
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
    process_type = (
        prepoperator.ProcessType.TRAINING
        if training
        else prepoperator.ProcessType.INFERENCE
    )
    tu_list = _run_pipeline(config_base, process_type, text)
    assert tu_list[0].src_detok == expected


def test_identity_filter():
    config = [
        {
            "op": "identity_filter",
            "verbose": True,
            "min_characters": 0,
        },
    ]

    assert _is_filtered(config, tu.TranslationUnit("Hello world!", "Hello world!"))
    assert not _is_filtered(config, tu.TranslationUnit("Hello world!", "Hello world"))
    config[0]["min_characters"] = 20
    assert not _is_filtered(config, tu.TranslationUnit("Hello world!", "Hello world!"))


@pytest.mark.parametrize(
    "filter_config,filtered",
    [
        (dict(), False),
        (dict(source=dict(max_characters=5)), True),
        (dict(target=dict(max_characters=5)), True),
        (dict(source=dict(max_characters=20)), False),
        (dict(source=dict(max_characters=20), target=dict(max_characters=15)), True),
        (dict(target=dict(max_characters=20, min_words=5)), True),
        (dict(source=dict(max_words=2)), True),
        (dict(min_words_ratio=1), True),
        (dict(max_words_ratio=0.5), True),
    ],
)
def test_length_filter(filter_config, filtered):
    filter_config.update({"op": "length_filter", "verbose": True})

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


def test_length_filter_empty_target():
    config = [
        {
            "op": "tokenization",
            "source": {"mode": "conservative", "joiner_annotate": True},
            "target": {"mode": "conservative", "joiner_annotate": True},
        },
        {
            "op": "length_filter",
            "min_words_ratio": 0.7,
            "max_words_ratio": 2,
        },
    ]
    source = "Hello"
    target = ""
    assert _is_filtered(config, tu.TranslationUnit(source, target))


@pytest.mark.parametrize(
    "mode,lower,upper",
    [
        ("hard_threshold", 0.5, None),
        ("hard_threshold", None, 0.5),
        ("hard_threshold", 3, 2),
        ("percent_threshold", 2, 0),
        ("percent_threshold", 0, 2),
        ("percent_threshold", 0.5, 0.6),
    ],
)
def test_align_perplexity_invalid_config(mode, lower, upper):
    config = {
        "source": "en",
        "target": "de",
        "preprocess": [
            {
                "op": "align_perplexity_filter",
                mode: {
                    "lower": lower,
                    "upper": upper,
                },
            }
        ],
    }
    with pytest.raises(ValueError, match="align_perplexity_filter"):
        prepoperator.Pipeline(config, prepoperator.ProcessType.TRAINING)


@pytest.mark.parametrize(
    "lower,upper,src_length,tgt_length,fwd_log_prob,bwd_log_prob,filtered",
    [
        (None, None, 5, 7, -5.1, -8.2, False),
        (-1, 0, 5, 5, 0, 0, False),
        (-10, -2, 5, 5, -5.1, -8.2, True),
    ],
)
def test_align_perplexity_hard_threshold(
    lower, upper, src_length, tgt_length, fwd_log_prob, bwd_log_prob, filtered
):
    config = [
        {
            "op": "align_perplexity_filter",
            "verbose": True,
            "hard_threshold": {
                "lower": lower,
                "upper": upper,
            },
        }
    ]

    tokenizer = pyonmttok.Tokenizer("conservative", joiner_annotate=True)
    single_tu = tu.TranslationUnit(
        " ".join(str(i) for i in range(src_length)),
        " ".join(str(i) for i in range(tgt_length)),
        source_tokenizer=tokenizer,
        target_tokenizer=tokenizer,
    )
    single_tu.set_alignment(
        _MockAligner(forward_log_prob=fwd_log_prob, backward_log_prob=bwd_log_prob)
    )
    assert filtered == _is_filtered(config, single_tu)


@pytest.mark.parametrize(
    "lower,upper,log_probs,expected_log_probs",
    [
        (0, 0, [-1.2, -3.5, -0.5, -7.0, -5.0, -1.3, -2.2, -5.8, -6.7, -18.1], None),
        (0.1, 0, [-1.2, -3.5, -0.5, -7.0, -5.0, -1.3, -5.8, -6.7, -18.1], None),
        (
            0.2,
            0.1,
            [-1.2, -3.5, -0.5, -7.0, -5.0, -1.3, -2.2, -5.8, -6.7, -18.1],
            [-1.2, -3.5, -5.0, -1.3, -2.2, -5.8, -6.7],
        ),
    ],
)
def test_align_perplexity_percent_threshold(
    lower, upper, log_probs, expected_log_probs
):
    if expected_log_probs is None:
        expected_log_probs = log_probs
    tu_list = []
    tokenizer = pyonmttok.Tokenizer("conservative", joiner_annotate=True)
    for log_prob in log_probs:
        single_tu = tu.TranslationUnit(
            "a b c", "a b c", source_tokenizer=tokenizer, target_tokenizer=tokenizer
        )
        single_tu.set_alignment(
            _MockAligner(forward_log_prob=log_prob, backward_log_prob=log_prob)
        )
        tu_list.append(single_tu)

    config = {
        "source": "en",
        "target": "fr",
        "preprocess": [
            {
                "op": "align_perplexity_filter",
                "verbose": True,
                "percent_threshold": {
                    "lower": lower,
                    "upper": upper,
                },
            }
        ],
    }

    tu_list = _run_pipeline(config, prepoperator.ProcessType.TRAINING, tu_list)
    assert len(tu_list) == len(expected_log_probs)
    for single_tu, log_prob in zip(tu_list, expected_log_probs):
        assert single_tu.alignment_log_probs[0][0] == log_prob


@pytest.mark.parametrize(
    "src,tgt,filtered,expected",
    [
        (
            "This is a test without parentheses.",
            "Ceci est un test sans parenthèses.",
            False,
            (None, None),
        ),
        (
            "This () is a test with one parenthesis in source.",
            "Ceci est un test avec une parenthèse en source.",
            False,
            ("This is a test with one parenthesis in source.", None),
        ),
        (
            "This is a test with one parenthesis in target.",
            "Ceci est un test avec (une parenthèse en cible)",
            False,
            (None, "Ceci est un test avec"),
        ),
        (
            "This is a test with one parenthesis (in source) and one in target.",
            "Ceci est un test avec une parenthèse en source (et une en cible).",
            False,
            (None, None),
        ),
        (
            "This is a test with (two) parentheses in source and (none) in target.",
            "Ceci est un test avec deux parenthèses en source et aucune en cible.",
            True,
            (None, None),
        ),
        (
            "(This is a test with two) parentheses in source and (two) in target.",
            "Ceci est un test avec (deux parenthèses en source) et (deux en cible).",
            False,
            (None, None),
        ),
        (
            "This is a test with (two) parentheses in source and (three in target)",
            "Ceci est un test avec (deux) parenthèses (en source) et (trois) en cible.",
            True,
            (None, None),
        ),
        (
            "This is a test with (mismatched> parentheses in source.",
            "Ceci est un test avec des parenthèses incorrectes en source.",
            True,
            (None, None),
        ),
        (
            "This is another test with (mismatched < parentheses in source.",
            "Ceci est un (autre) test avec des parenthèses incorrectes en source.",
            True,
            (None, None),
        ),
        (
            "This a yest another test with (mismatched parentheses in source.",
            "Ceci est <encore un autre> test avec des (parenthèses) incorrectes en source.",
            True,
            (None, None),
        ),
        (
            "This is a (test (with nested) parentheses) in source.",
            "Ceci est un test avec des parenthèses imbriquées en source.",
            True,
            (None, None),
        ),
        (
            "This is another <test (with nested) parentheses> in source.",
            "Ceci est un <autre> test avec <des parenthèses imbriquées> en source.",
            True,
            (None, None),
        ),
        (
            "This is a test with (mismatched) parentheses in target.",
            "Ceci est un test avec des (parenthèses (incorrectes en cible.",
            True,
            (None, None),
        ),
        (
            "This is another test with (mismatched) parentheses in target.",
            "Ceci est un autre test avec des <parenthèses <incorrectes en cible.",
            True,
            (None, None),
        ),
        (
            "This a yest another test with <mismatched> parentheses in target.",
            "Ceci est encore un autre test avec des <parenthèses incorrectes) en cible.",
            True,
            (None, None),
        ),
        (
            "This is a (test) (with nested) (parentheses) in target.",
            "Ceci est un <test avec <des parenthèses imbriquées> en cible>.",
            True,
            (None, None),
        ),
        (
            "This is another <test (with nested) parentheses> in target.",
            "Ceci est un autre <test avec (des parenthèses imbriquées) en cible>.",
            True,
            (None, None),
        ),
        (
            "<This is> a more complicated (test) not to filter (and not to) modify.",
            "(Ceci est) un test plus compliqué à ne pas <filter> ni (modifier)",
            False,
            (None, None),
        ),
        (
            "(This is) a more complicated (test) <to modify>.",
            "(Ceci) est (un test) plus compliqué à modifier.",
            False,
            ("(This is) a more complicated (test).", None),
        ),
        (
            "(And) yet another test to modify.",
            "Et encore un test <à modifier>",
            False,
            ("yet another test to modify.", "Et encore un test"),
        ),
        (
            "Test for(joiner)replacement.",
            "Teste pour remplacer le joiner.",
            False,
            ("Test forreplacement.", None),
        ),
        (
            "Test for joiner replacement.",
            "Teste pour remplacer(le)joiner.",
            False,
            (None, "Teste pour remplacerjoiner."),
        ),
    ],
)
def test_parentheses_filter(src, tgt, filtered, expected):
    config = [
        {
            "op": "tokenization",
            "source": {"mode": "conservative", "joiner_annotate": True},
            "target": {"mode": "conservative", "joiner_annotate": True},
        },
        {"op": "parentheses", "side": "both", "type": [["(", ")"], ["<", ">"]]},
    ]

    TU = tu.TranslationUnit(src, tgt)
    assert filtered == _is_filtered(config, TU)
    if not filtered:
        result_src = TU.src_detok
        result_tgt = TU.tgt_detok
        if expected[0] is None:
            assert src == result_src
        else:
            assert expected[0] == result_src
        if expected[1] is None:
            assert tgt == result_tgt
        else:
            assert expected[1] == result_tgt
