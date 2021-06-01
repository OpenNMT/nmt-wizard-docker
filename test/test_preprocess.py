# -*- coding: utf-8 -*-

import pytest
import shutil
import os
from copy import deepcopy
import random

from nmtwizard import utils
from nmtwizard.preprocess.consumer import Consumer
from nmtwizard.preprocess.loader import Loader
from nmtwizard.preprocess.preprocess import (
    Processor,
    InferenceProcessor,
    TrainingProcessor,
)
from nmtwizard.preprocess.tokenizer import vocabulary_iterator
from nmtwizard.preprocess.tu import TranslationUnit
from nmtwizard.preprocess import prepoperator


def generate_pseudo_corpus(corpus_dir, size, name, suffix):
    path = str(corpus_dir.join(name + "." + suffix))
    with utils.open_file(path, "wb") as f:
        for i in range(size):
            f.write((name + " " + str(i) + "\n").encode("utf-8"))
    return path


@pytest.mark.parametrize("batch_size,num_threads", [(10, 1), (10, 2), (10000, 1)])
def test_sampler(tmpdir, batch_size, num_threads):
    os.environ["NB_CPU"] = str(num_threads)

    corpus_dir = tmpdir.join("corpus")
    corpus_dir.mkdir()

    generate_pseudo_corpus(corpus_dir, 800, "corpus_specific1", "en")
    generate_pseudo_corpus(corpus_dir, 800, "corpus_specific1", "de")
    generate_pseudo_corpus(corpus_dir, 2000, "corpus_specific2", "en")
    generate_pseudo_corpus(corpus_dir, 2000, "corpus_specific2", "de")
    generate_pseudo_corpus(corpus_dir, 50, "generic_added", "en")
    generate_pseudo_corpus(corpus_dir, 50, "generic_added", "de")
    generate_pseudo_corpus(corpus_dir, 100, "generic_corpus", "en")
    generate_pseudo_corpus(corpus_dir, 100, "generic_corpus", "de")
    generate_pseudo_corpus(corpus_dir, 200, "IT", "en")
    generate_pseudo_corpus(corpus_dir, 200, "IT", "de")
    generate_pseudo_corpus(corpus_dir, 3000, "news_pattern", "en")
    generate_pseudo_corpus(corpus_dir, 3000, "news_pattern", "de")
    generate_pseudo_corpus(corpus_dir, 10, "unaligned", "en")
    generate_pseudo_corpus(corpus_dir, 10, "generic_to_ignore", "en")
    generate_pseudo_corpus(corpus_dir, 10, "generic_to_ignore", "de")

    config = {
        "source": "en",
        "target": "de",
        "data": {
            "batch_size": batch_size,
            "sample": 5000,
            "sample_dist": [
                {
                    "path": str(corpus_dir),
                    "distribution": [
                        ["generic_to_ignore", 0],
                        ["generic", 1],
                        ["specific", 5.2],
                        ["news.*pattern", "*1"],
                        [".*something", 1],
                        ["unaligned", 1],
                    ],
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
    }

    preprocessor = TrainingProcessor(config, "", str(tmpdir))
    (
        data_path,
        train_dir,
        num_samples,
        summary,
        _,
    ) = preprocessor.generate_preprocessed_data()
    assert num_samples == 5000
    assert summary["news_pattern"]["linesampled"] == 3000
    assert summary["generic_corpus"]["linesampled"] >= 215
    assert summary["generic_added"]["linesampled"] >= 107
    assert summary["corpus_specific1"]["linesampled"] >= 479
    assert summary["corpus_specific2"]["linesampled"] >= 1198
    assert "IT" not in summary
    assert "unaligned" not in summary
    assert "generic_to_ignore" not in summary

    # check unique sampling with undersampling
    with open(str(tmpdir.join("preprocess/corpus_specific2.en")), "rb") as f:
        rf = f.readlines()
        rf_list = [line.strip() for line in rf]
        assert len(rf_list) == len(set(rf_list))

    # check unique sampling with oversampling
    with open(str(tmpdir.join("preprocess/generic_added.en")), "rb") as f:
        rf_dict = {}
        for line in f:
            if line in rf_dict:
                rf_dict[line] += 1
            else:
                rf_dict[line] = 1
        for c in rf_dict.values():
            assert 2 <= c <= 3

    # Check strict mode
    shutil.rmtree(str(tmpdir.join("preprocess")))
    config["data"]["sample_dist"][0]["mode_strict"] = True

    preprocessor = TrainingProcessor(config, "", str(tmpdir))
    with pytest.raises(RuntimeError) as excinfo:
        (
            data_path,
            train_dir,
            num_samples,
            summary,
            _,
        ) = preprocessor.generate_preprocessed_data()
    assert (
        str(excinfo.value)
        == "pattern '.*something' in block 0 doesn't match any file with strict mode enabled."
    )

    shutil.rmtree(str(tmpdir.join("preprocess")))
    config["data"]["sample_dist"] = [
        {
            "path": str(corpus_dir),
            "distribution": [
                ["generic", 1],
                ["specific", 5.2],
                ["news.*pattern", "*2"],
                [".*something", 1],
            ],
        }
    ]

    preprocessor = TrainingProcessor(config, "", str(tmpdir))
    (
        data_path,
        train_dir,
        num_samples,
        summary,
        _,
    ) = preprocessor.generate_preprocessed_data()

    assert num_samples == 6000
    assert summary["news_pattern"]["linesampled"] == 6000
    assert summary["generic_corpus"]["linesampled"] == 0
    assert summary["generic_added"]["linesampled"] == 0
    assert summary["corpus_specific1"]["linesampled"] == 0
    assert summary["corpus_specific2"]["linesampled"] == 0
    assert "IT" not in summary

    # Test oversampling as example weights
    shutil.rmtree(str(tmpdir.join("preprocess")))
    config["data"]["oversample_with_sentence_weighting"] = True

    preprocessor = TrainingProcessor(config, "", str(tmpdir))
    (
        data_path,
        train_dir,
        num_samples,
        summary,
        _,
    ) = preprocessor.generate_preprocessed_data()

    assert summary["news_pattern"]["linesampled"] == 3000
    with open(str(tmpdir.join("preprocess/news_pattern.weights")), "r") as f:
        rf = f.readlines()
        assert len(rf) == 3000
        assert all(el == "2.0\n" for el in rf)

    shutil.rmtree(str(tmpdir.join("preprocess")))
    config["data"]["sample_dist"] = [
        {
            "path": str(corpus_dir),
            "distribution": [["generic_to_ignore", 0], ["generic", "*"]],
        }
    ]

    preprocessor = TrainingProcessor(config, "", str(tmpdir))
    (
        data_path,
        train_dir,
        num_samples,
        summary,
        _,
    ) = preprocessor.generate_preprocessed_data()

    assert num_samples == 150
    assert summary["generic_corpus"]["linesampled"] == 100
    assert summary["generic_added"]["linesampled"] == 50

    del os.environ["NB_CPU"]


@pytest.mark.parametrize(
    "similarity_filter_config,expected_num_samples",
    [
        (dict(mode="hard"), 4),
        (dict(mode="soft_linear"), None),  # Just check that it runs without error.
        (dict(mode="soft_sigmoid"), None),  # Same.
    ],
)
def test_sampler_with_annotations(
    tmpdir, similarity_filter_config, expected_num_samples
):
    similarity_filter_config.update({"op": "similarity_filter", "verbose": True})

    with open(str(tmpdir.join("train.en")), "w") as en:
        en.write("\n".join(["1", "2", "3", "4", "5", "6"]))
    with open(str(tmpdir.join("train.fr")), "w") as fr:
        fr.write("\n".join(["1", "2", "3", "4", "5", "6"]))

    annot_dir = tmpdir.join("train_enfr_annot")
    os.makedirs(str(annot_dir))
    with open(str(annot_dir.join("train")), "w") as annot:
        annot.write(
            "\n".join(["0.0274", "-0.1201", "0.2499", "0.8566", "-0.8025", "0.0892"])
        )

    from_dir = str(tmpdir)
    to_dir = str(tmpdir.join("output"))
    os.makedirs(to_dir)

    config = {
        "source": "en",
        "target": "fr",
        "data": {
            "sample_dist": [
                {
                    "path": from_dir,
                    "distribution": [["train", "*"]],
                    "annotations": {"similarity": "train_enfr_annot"},
                }
            ]
        },
        "preprocess": [similarity_filter_config],
    }

    preprocessor = TrainingProcessor(config, from_dir, to_dir)
    (
        data_path,
        train_dir,
        num_samples,
        summary,
        _,
    ) = preprocessor.generate_preprocessed_data()

    assert "annotations" in summary["train"] and summary["train"]["annotations"] == [
        "similarity"
    ]
    if expected_num_samples is not None:
        assert num_samples == expected_num_samples


# TODO : test generate vocabularies with several tokenizations
def _test_generate_vocabularies(
    tmpdir, size, min_frequency, real_size, subword_config=None, multi=False
):

    sides = {}
    if multi:
        sides["multi"] = "en_de"
    else:
        sides["source"] = "en"
        sides["target"] = "de"

    config = {
        "source": "en",
        "target": "de",
        "data": {
            "sample": 800,
            "sample_dist": [
                {
                    "path": os.path.join(
                        os.path.dirname(os.path.realpath(__file__)), "corpus", "train"
                    ),
                    "distribution": [["europarl", 1]],
                }
            ],
        },
        "preprocess": [
            {
                "op": "tokenization",
                "source": {"mode": "aggressive"},
                "target": {"mode": "aggressive"},
                "multi": {},
            }
        ],
    }

    for side, ext in sides.items():
        config["preprocess"][0][side]["build_vocabulary"] = {
            "name": "test",
            "size": size,
            "min-frequency": min_frequency,
            "add": ["mama", "papa"],
            "merge": os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "corpus",
                "vocab",
                "vocab-extra.txt",
            ),
        }
        config["preprocess"][0][side]["build_subword"] = subword_config

    preprocessor = TrainingProcessor(config, "", str(tmpdir))
    (
        _,
        result_preprocess_config,
        result_vocab_config,
    ) = preprocessor.generate_vocabularies()

    for side, ext in sides.items():

        # Check subword
        if subword_config:
            subword_type = subword_config["type"]
            subword_file_name = "subword/"
            if multi:
                subword_file_name += "joint_"
            subword_file_name += "%s_model0-100.%s" % (subword_type, ext)
            subword_file = str(tmpdir.join(subword_file_name))

            if subword_type == "bpe":
                with open(subword_file, "rb") as f:
                    assert len(f.readlines()) == 101

            if side == "multi":
                assert (
                    result_preprocess_config[0]["source"][
                        "%s_model_path" % subword_type
                    ]
                    == subword_file
                )
                assert (
                    result_preprocess_config[0]["target"][
                        "%s_model_path" % subword_type
                    ]
                    == subword_file
                )
            else:
                assert (
                    result_preprocess_config[0][side]["%s_model_path" % subword_type]
                    == subword_file
                )

        # Check vocabulary
        rs = real_size[side] if isinstance(real_size, dict) else real_size
        header_rs = rs + 2  # for headers

        vocab_file_name = "vocabulary/"
        if multi:
            vocab_file_name += "joint_"
        vocab_file_name += "vocab_test-%s.%s" % (rs, ext)
        vocab_file = str(tmpdir.join(vocab_file_name))

        with open(vocab_file, "rb") as f:
            assert len(f.readlines()) == header_rs

        if side == "multi":
            assert result_vocab_config["source"]["path"] == vocab_file
            assert result_vocab_config["target"]["path"] == vocab_file
        else:
            assert result_vocab_config[side]["path"] == vocab_file


def test_generate_vocabularies(tmpdir):

    # Real vocabulary is smaller than size
    _test_generate_vocabularies(tmpdir, 2000, 0, {"source": 1432, "target": 1704})

    # Real vocabulary is greater than size
    _test_generate_vocabularies(tmpdir, 1000, 0, 1000)

    # Size is greater than filtering by frequency
    _test_generate_vocabularies(tmpdir, 1000, 5, {"source": 636, "target": 618})

    # Size is smaller than filtering by frequency
    _test_generate_vocabularies(tmpdir, 100, 5, 100)

    config_subword_bpe = {"params": {"vocab_size": 100}, "type": "bpe"}

    _test_generate_vocabularies(tmpdir, 50, 5, 50, config_subword_bpe)

    config_subword_sp = {"params": {"vocab_size": 100}, "type": "sp"}

    _test_generate_vocabularies(tmpdir, 50, 5, 50, config_subword_sp)

    _test_generate_vocabularies(tmpdir, 50, 5, 50, config_subword_bpe, True)


def test_op_adding_tokens(tmpdir):
    @prepoperator.register_operator("op_adding_tokens")
    class OpAddingTokens(prepoperator.TUOperator):
        def _preprocess_tu(self, tu, meta_batch):
            meta_batch["tokens_to_add"] = {"source": ["a", "b", "b"], "target": ["c"]}
            return [tu]

    corpus_dir = tmpdir.join("corpus")
    corpus_dir.mkdir()

    generate_pseudo_corpus(corpus_dir, 100, "generic_corpus", "en")
    generate_pseudo_corpus(corpus_dir, 100, "generic_corpus", "de")

    config = {
        "source": "en",
        "target": "de",
        "data": {
            "sample": 0,
            "sample_dist": [
                {
                    "path": str(corpus_dir),
                    "distribution": [
                        ["generic", "*"],
                    ],
                }
            ],
        },
        "preprocess": [
            {
                "op": "op_adding_tokens",
            },
            {
                "op": "tokenization",
                "source": {"mode": "space", "build_vocabulary": {"size": 20}},
                "target": {"mode": "space", "build_vocabulary": {"size": 20}},
            },
        ],
    }

    preprocessor = TrainingProcessor(config, "", str(tmpdir))
    _, _, vocab_config = preprocessor.generate_vocabularies()
    source_vocabulary = set(vocabulary_iterator(vocab_config["source"]["path"]))
    target_vocabulary = set(vocabulary_iterator(vocab_config["target"]["path"]))
    assert source_vocabulary > set(["a", "b"])
    assert target_vocabulary > set(["c"])

    _, _, _, _, tokens_to_add = preprocessor.generate_preprocessed_data()
    assert set(tokens_to_add["source"]) == set(["a", "b"])
    assert set(tokens_to_add["target"]) == set(["c"])


def test_preprocess_pipeline(tmpdir):

    corpus_dir = tmpdir.join("corpus")
    corpus_dir.mkdir()

    config = {
        "source": "en",
        "target": "de",
        "data": {
            "sample": 800,
            "sample_dist": [
                {
                    "path": str(corpus_dir),
                    "distribution": [
                        ["extra_generic", 1, "generic_label"],
                        ["generic", 1, "generic_label"],
                        ["good_news", 1, "news_label"],
                        ["news", 1, "news_label"],
                        ["no_label", 1],
                    ],
                }
            ],
        },
        "preprocess": [
            {
                "op": "tokenization",
                "source": {"mode": "aggressive", "joiner_annotate": True},
                "target": {"mode": "aggressive", "joiner_annotate": True},
            },
            {
                "op": "length_filter",
                "source": {"max_characters": 3},
                "overrides": {
                    "generic_label": {"disabled": True},
                    "news_label": {"source": {"max_characters": 6}},
                },
            },
        ],
    }

    generate_pseudo_corpus(corpus_dir, 10, "generic", "en")
    generate_pseudo_corpus(corpus_dir, 10, "generic", "de")
    generate_pseudo_corpus(corpus_dir, 10, "extra_generic", "en")
    generate_pseudo_corpus(corpus_dir, 10, "extra_generic", "de")
    generate_pseudo_corpus(corpus_dir, 10, "good_news", "en")
    generate_pseudo_corpus(corpus_dir, 10, "good_news", "de")
    generate_pseudo_corpus(corpus_dir, 10, "news", "en")
    generate_pseudo_corpus(corpus_dir, 10, "news", "de")
    generate_pseudo_corpus(corpus_dir, 10, "no_label", "en")
    generate_pseudo_corpus(corpus_dir, 10, "no_label", "de")
    generate_pseudo_corpus(corpus_dir, 10, "generic_news", "en")
    generate_pseudo_corpus(corpus_dir, 10, "generic_news", "de")

    preprocessor = TrainingProcessor(config, "", str(tmpdir))
    (
        data_path,
        train_dir,
        num_samples,
        summary,
        _,
    ) = preprocessor.generate_preprocessed_data()

    for f_name, f_info in summary.items():
        if "generic" in f_name or f_name == "news":
            # generic is not filtered because filtering is disabled
            # 'news' is not filtered because length is overriden
            assert f_info["linefiltered"] == f_info["linesampled"]
        else:
            assert f_info["linefiltered"] == 0

    # Test multiple override labels
    shutil.rmtree(str(tmpdir.join("preprocess")))
    config["data"]["sample_dist"][0]["distribution"].insert(
        0, ["generic_news", 1, ["generic_label", "news_label"]]
    )

    preprocessor = TrainingProcessor(config, "", str(tmpdir))
    with pytest.raises(RuntimeError) as excinfo:
        (
            data_path,
            train_dir,
            num_samples,
            summary,
            _,
        ) = preprocessor.generate_preprocessed_data()
    assert str(excinfo.value).startswith("One corpus requires different overrides")

    config["preprocess"] = [
        {
            "op": "tokenization",
            "source": {"mode": "aggressive", "joiner_annotate": True},
            "target": {"mode": "aggressive", "joiner_annotate": True},
            "overrides": {"news_label": {"disabled": True}},
        },
        {
            "op": "length_filter",
            "source": {"max_characters": 3},
            "overrides": {"generic_label": {"disabled": True}},
        },
    ]

    preprocessor = TrainingProcessor(config, "", str(tmpdir))
    (
        data_path,
        train_dir,
        num_samples,
        summary,
        _,
    ) = preprocessor.generate_preprocessed_data()

    for f_name, f_info in summary.items():
        if "generic" in f_name:
            # generic is not filtered because filtering is disabled
            assert f_info["linefiltered"] == f_info["linesampled"]
        else:
            assert f_info["linefiltered"] == 0

    with open(str(tmpdir.join("preprocess/generic_news.en"))) as f:
        # tokenization is disabled for news label
        assert f.readline().startswith("generic_news")

    with open(str(tmpdir.join("preprocess/extra_generic.en"))) as f:
        # tokenization is enabled for other corpora (only generic label, no news label)
        assert f.readline().startswith("extra ￭_￭ generic")

    prep = InferenceProcessor(config)
    source, target, _ = prep.process_input("This is a test.")
    assert source[0] == ["This", "is", "a", "test", "￭."]  # First and only part.
    assert target[0] is None

    source2, target, _ = prep.process_input("This is a test.", "Das ist...")
    assert source2 == source
    assert target[0] == ["Das", "ist", "￭.", "￭.", "￭."]

    post = InferenceProcessor(config, postprocess=True)
    assert post.process_input(source, target) == "Das ist..."


config_base = {
    "source": "en",
    "target": "de",
    "data": {
        "sample": 800,
        "sample_dist": [
            {
                "path": os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "corpus", "train"
                ),
                "distribution": [["europarl", 1]],
            }
        ],
    },
    "preprocess": [
        {
            "op": "tokenization",
            "source": {"mode": "aggressive", "joiner_annotate": True},
            "target": {"mode": "aggressive", "joiner_annotate": True},
        },
        {
            "op": "alignment",
            "write_alignment": True,
            "forward": {
                "probs": os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "corpus",
                    "resources",
                    "alignment",
                    "ende_forward.probs",
                )
            },
            "backward": {
                "probs": os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "corpus",
                    "resources",
                    "alignment",
                    "ende_backward.probs",
                )
            },
        },
    ],
}


def test_pipeline_with_options():
    @prepoperator.register_operator("dummy_op_without_options")
    class DummyOpWithoutOptions(prepoperator.TUOperator):
        def _preprocess_tu(self, tu, meta_batch):
            return [tu]

    @prepoperator.register_operator("dummy_op_with_options")
    class DummyOpWithOptions(prepoperator.TUOperator):
        @staticmethod
        def accept_options():
            return True

        def _preprocess_tu(self, tu, meta_batch, options=None):
            assert isinstance(options, dict)
            return [tu]

    config = {
        "source": "en",
        "target": "fr",
        "preprocess": [
            {
                "op": "tokenization",
                "source": {"mode": "conservative", "joiner_annotate": True},
                "target": {"mode": "conservative", "joiner_annotate": True},
            },
            {
                "op": "dummy_op_without_options",
                "name": "wo_opts",
            },
            {
                "op": "dummy_op_with_options",
                "name": "w_opts",
            },
        ],
    }

    processor = InferenceProcessor(config)

    options = {"wo_opts": {"my-option": 42}}
    with pytest.raises(RuntimeError):
        processor.process_input("Hello", options=options)

    options = {"w_opts": {"my-option": 42}}
    processor.process_input("Hello", options=options)


def test_preprocess_gzip_file(tmpdir):
    num_lines = 10
    input_path = generate_pseudo_corpus(tmpdir, num_lines, "input", "en.gz")
    processor = InferenceProcessor(config_base)
    output_path, _ = processor.process_file(input_path)

    assert os.path.basename(output_path) == "input.en.tok"
    assert utils.count_lines(output_path)[1] == num_lines


def test_preprocess_empty_line(tmpdir):
    processor = InferenceProcessor(config_base)
    source, _, _ = processor.process_input("")
    assert source[0] == []

    input_path = str(tmpdir.join("empty.txt"))
    with open(input_path, "w") as input_file:
        input_file.write("\n")
    output_path, _ = processor.process_file(input_path)
    with open(output_path) as output_file:
        assert output_file.readlines() == ["\n"]


def test_postprocess_multipart_file_loader(tmpdir):
    src_num_lines = 8
    src_input_path = generate_pseudo_corpus(tmpdir, src_num_lines, "input", "en")
    tgt_num_lines = 8
    tgt_input_path = generate_pseudo_corpus(tmpdir, tgt_num_lines, "input", "de")
    processor = InferenceProcessor(config_base, postprocess=True)

    meta = [
        [None, None, None],
        [None, None],
        [None],
        [None, None],
    ]

    output_path = processor.process_file(src_input_path, tgt_input_path, meta)

    assert os.path.basename(output_path) == "input.de.detok"
    assert utils.count_lines(output_path)[1] == 4


def test_postprocess_multipart_batch_loader(tmpdir):
    processor = InferenceProcessor(config_base, postprocess=True)

    source = [["Hello"], ["world"]]
    target = [["Bonjour"], ["monde"]]
    metadata = [None, None]

    target = processor.process_input(source, target, metadata=metadata)
    assert target == "Bonjour monde"


@pytest.mark.parametrize("num_cpus", [1, 2])
def test_preprocess_align(tmpdir, num_cpus):
    os.environ["NB_CPU"] = str(num_cpus)

    preprocessor = TrainingProcessor(config_base, "", str(tmpdir))
    (
        data_path,
        train_dir,
        num_samples,
        summary,
        _,
    ) = preprocessor.generate_preprocessed_data()

    with open(
        os.path.join(str(tmpdir), "preprocess", "europarl-v7.de-en.10K.tok.align")
    ) as align:
        assert align.readline().strip() == "0-0 1-0 2-1 3-2"

    del os.environ["NB_CPU"]


def test_replace_tokens(tmpdir):

    # Dummy token replacement operator.
    @prepoperator.register_operator("repl")
    class ReplaceOperator(prepoperator.TUOperator):
        def __init__(self, config, process_type, build_state):
            self._rand_repl_gen = self._replacement_generator()

        @staticmethod
        def is_stateful():
            return False

        @staticmethod
        def _replacement_generator():
            src_len = 0
            tgt_len = 0

            replacement_tokens = ["TEST1", "TEST2", "TEST3"]
            while True:
                src_len, tgt_len = yield

                src_pos = random.randint(0, src_len)
                tgt_pos = random.randint(0, tgt_len)

                src_num_to_del = random.randint(0, src_len - src_pos)
                tgt_num_to_del = random.randint(0, tgt_len - tgt_pos)

                src_tok_replace = replacement_tokens[
                    0 : random.randint(0, len(replacement_tokens))
                ]
                tgt_tok_replace = replacement_tokens[
                    0 : random.randint(0, len(replacement_tokens))
                ]
                yield (
                    (src_pos, src_num_to_del, src_tok_replace),
                    (tgt_pos, tgt_num_to_del, tgt_tok_replace),
                )

        def _preprocess_tu(self, tu, training):
            def checks_side(tokens, length, pos, num_to_del, tok_replace):
                assert len(tokens[0]) == length - num_to_del + len(tok_replace)

                if tok_replace:
                    assert tokens[0][pos : pos + len(tok_replace)] == tok_replace

            def change_align_side(al_idx, pos, num_to_del, tok_replace):
                new_al_idx = al_idx
                len_tok_replace = len(tok_replace) if tok_replace else 0
                if al_idx >= pos:
                    if al_idx < pos + num_to_del:
                        # insertion
                        if len_tok_replace:
                            new_al_idx = pos
                        # deletion
                        else:
                            return None
                    else:
                        # shift
                        new_al_idx = al_idx - num_to_del + len_tok_replace
                return new_al_idx

            if tu.src_tok.tokens and tu.tgt_tok.tokens:
                alignment_before = deepcopy(tu.alignment[0])

                next(self._rand_repl_gen)
                src_len = len(tu.src_tok.tokens[0])
                tgt_len = len(tu.tgt_tok.tokens[0])
                src_replace, tgt_replace = self._rand_repl_gen.send((src_len, tgt_len))

                src_pos, src_num_to_del, src_tok_replace = src_replace
                tgt_pos, tgt_num_to_del, tgt_tok_replace = tgt_replace

                tu.replace_tokens(src_replace, tgt_replace)

                checks_side(
                    tu.src_tok.tokens, src_len, src_pos, src_num_to_del, src_tok_replace
                )
                checks_side(
                    tu.tgt_tok.tokens, tgt_len, tgt_pos, tgt_num_to_del, tgt_tok_replace
                )

                # Check alignment
                alignment_after = tu.alignment[0]

                new_align = set()
                for al_src, al_tgt in alignment_before:
                    new_al_src = change_align_side(
                        al_src, src_pos, src_num_to_del, src_tok_replace
                    )
                    if new_al_src is None:
                        continue
                    new_al_tgt = change_align_side(
                        al_tgt, tgt_pos, tgt_num_to_del, tgt_tok_replace
                    )
                    if new_al_tgt is None:
                        continue
                    new_align.add((new_al_src, new_al_tgt))

                if src_tok_replace and tgt_tok_replace:
                    src_nb_inserted_tok = len(src_tok_replace)
                    tgt_nb_inserted_tok = len(tgt_tok_replace)
                    for i in range(src_nb_inserted_tok):
                        new_align.add((src_pos + i, tgt_pos))
                    if tgt_nb_inserted_tok > 1:
                        new_align.add(
                            (
                                src_pos + src_nb_inserted_tok - 1,
                                tgt_pos + tgt_nb_inserted_tok - 1,
                            )
                        )

                assert new_align == alignment_after
            return [tu]

    config_replace = deepcopy(config_base)

    config_replace["preprocess"].append({"op": "repl"})

    preprocessor = TrainingProcessor(config_replace, "", str(tmpdir))
    (
        data_path,
        train_dir,
        num_samples,
        summary,
        _,
    ) = preprocessor.generate_preprocessed_data()


def test_extra_target(tmpdir):

    # Dummy operator that inserts extra target.
    @prepoperator.register_operator("extra_target")
    class ExtraTargetOperator(prepoperator.TUOperator):
        def is_applied_for(process_type):
            return process_type != prepoperator.ProcessType.POSTPROCESS

        def _preprocess_tu(self, tu, training):
            tu.add_target("Das ist ein neues Ziel.", "extra")
            assert tu.has_target("extra")
            tu.set_target_output("extra", side="source", delimiter="｟delimiter_token｠")
            return [tu]

    config_extra_target = deepcopy(config_base)

    config_extra_target["preprocess"].insert(0, {"op": "extra_target"})

    preprocessor = TrainingProcessor(config_extra_target, "", str(tmpdir))
    (
        data_path,
        train_dir,
        num_samples,
        summary,
        _,
    ) = preprocessor.generate_preprocessed_data()

    with open(str(tmpdir.join("preprocess/europarl-v7.de-en.10K.tok.en"))) as f:
        for line in f:
            assert line.strip().endswith(
                "｟delimiter_token｠ Das ist ein neues Ziel ￭."
            ) or line.strip().endswith("Das ist ein neues Ziel ￭.")

    preprocessor = InferenceProcessor(config_extra_target)
    source, target, _ = preprocessor.process_input("This is a test.")
    assert source[0] == [
        "This",
        "is",
        "a",
        "test",
        "￭.",
        "｟delimiter_token｠",
        "Das",
        "ist",
        "ein",
        "neues",
        "Ziel",
        "￭.",
    ]
    assert target[0] is None

    target = [["Das", "ist", "ein", "Test", "￭."]]
    post = InferenceProcessor(config_extra_target, postprocess=True)
    assert post.process_input(source, target) == "Das ist ein Test."


@pytest.mark.parametrize("num_workers", [0, 2])
def test_shared_state_with_overrides(num_workers):
    class CustomLoader(Loader):
        def __init__(self, labels):
            super().__init__(batch_size=1)
            self._labels = labels

        def __call__(self):
            for label in self._labels:
                yield [TranslationUnit("")], {"label": label}

    class CustomConsumer(Consumer):
        def _consume(self, tu_batch):
            pass

    class SharedClass:
        def __init__(self, value):
            self._value = value

        def value(self):
            return self._value

    op_name = "op_with_shared_state_%d" % num_workers

    @prepoperator.register_operator(op_name)
    class OpWithSharedState(prepoperator.Operator):

        @classmethod
        def _config_schema(cls):
            schema = super(OpWithSharedState, cls)._config_schema()

            schema["properties"].update({"value": {"type": "string"}})
            return schema

        @staticmethod
        def get_shared_classes():
            return [SharedClass]

        @staticmethod
        def get_shared_builders(config, process_type):
            return {"shared_obj": (SharedClass, (config["value"],))}

        def __init__(self, config, process_type, build_state, shared_state):
            assert len(shared_state) == 1
            obj = shared_state["shared_obj"]
            assert obj.value() == config["value"]
            if num_workers == 0:
                assert isinstance(obj, SharedClass)
            else:
                import multiprocessing.managers

                assert isinstance(obj, multiprocessing.managers.BaseProxy)

        def _preprocess(self, tu_batch, **kwargs):
            return tu_batch

    config = {
        "source": "en",
        "target": "fr",
        "preprocess": [
            {
                "op": op_name,
                "value": "one",
                "overrides": {
                    "two": {
                        "value": "two",
                    },
                },
            },
        ],
    }

    processor = Processor(
        config, prepoperator.ProcessType.TRAINING, num_workers=num_workers
    )
    loader = CustomLoader([None, "two", "one", None])
    consumer = CustomConsumer()

    processor.process(loader, consumer)
