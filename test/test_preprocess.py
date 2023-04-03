import pytest
import shutil
import os
from copy import deepcopy
import random
import glob
import time
import requests_mock
import itertools
import json

from nmtwizard import beat_service
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


def generate_lines(size, name):
    for i in range(size):
        yield "%s %d" % (name, i)


def generate_pseudo_corpus(corpus_dir, size, name, suffix):
    path = str(corpus_dir.join(name + "." + suffix))
    with utils.open_file(path, "wt") as f:
        for line in generate_lines(size, name):
            f.write("%s\n" % line)
    return path


def generate_parallel_corpus(corpus_dir, size, name, src, tgt, corpus_format="bitext"):
    if corpus_format == "json":
        generate_json_corpus(corpus_dir, size, name, src, tgt)
    else:
        generate_bitext_corpus(corpus_dir, size, name, src, tgt)


def generate_bitext_corpus(corpus_dir, size, name, src, tgt):
    generate_pseudo_corpus(corpus_dir, size, name, src)
    generate_pseudo_corpus(corpus_dir, size, name, tgt)


def generate_json_corpus(corpus_dir, size, name, src, tgt):
    path = str(corpus_dir.join(name))
    src_lines = generate_lines(size, name)
    tgt_lines = generate_lines(size, name)
    return save_corpus_as_json(path, src, tgt, zip(src_lines, tgt_lines))


def save_corpus_as_json(path, src, tgt, segments):
    segments = list(segments)

    dirname, basename = os.path.split(path)
    basename = os.path.splitext(basename)[0]

    metadata_path = os.path.join(dirname, ".%s.metadata" % basename)
    metadata = {"files": [{"nbSegments": len(segments)}]}
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file)

    all_segments = []

    for fields in segments:
        src_line = fields[0]
        tgt_line = fields[1]
        extra_fields = {"metadata": fields[2]} if len(fields) > 2 else {}

        segment = {
            "language": src,
            "seg": src_line,
            "tgts": [
                {
                    "language": tgt,
                    "seg": tgt_line,
                    **extra_fields,
                }
            ],
        }

        all_segments.append(segment)

    data = {"bidirectional": False, "segments": all_segments}
    path = path + ".json"

    with open(path, "w") as json_file:
        json.dump(data, json_file)

    return path


@pytest.mark.parametrize("batch_size,num_threads", [(10, 1), (10, 2), (10000, 1)])
@pytest.mark.parametrize("corpus_format", ["bitext", "json"])
def test_sampler(tmpdir, batch_size, num_threads, corpus_format):
    os.environ["NB_CPU"] = str(num_threads)

    corpus_dir = tmpdir.join("corpus")
    corpus_dir.mkdir()

    generate_parallel_corpus(
        corpus_dir, 800, "corpus_specific1", "en", "de", corpus_format
    )
    generate_parallel_corpus(
        corpus_dir, 2000, "corpus_specific2", "en", "de", corpus_format
    )
    generate_parallel_corpus(corpus_dir, 50, "generic_added", "en", "de", corpus_format)
    generate_parallel_corpus(
        corpus_dir, 100, "generic_corpus", "en", "de", corpus_format
    )
    generate_parallel_corpus(corpus_dir, 200, "IT", "en", "de", corpus_format)
    generate_parallel_corpus(
        corpus_dir, 3000, "news_pattern", "en", "de", corpus_format
    )
    generate_parallel_corpus(
        corpus_dir, 10, "generic_to_ignore", "en", "de", corpus_format
    )

    if corpus_format == "bitext":
        generate_pseudo_corpus(corpus_dir, 10, "unaligned", "en")

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

    weights_file_list = glob.glob(str(tmpdir.join("preprocess/*.weights")))
    assert len(weights_file_list) == 0

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

    config["data"]["sample_dist"][0]["distribution"].insert(0, ["specific", 1])
    preprocessor = TrainingProcessor(config, "", str(tmpdir))
    with pytest.raises(RuntimeError) as excinfo:
        preprocessor.generate_preprocessed_data()
    assert (
        str(excinfo.value)
        == "pattern 'specific' in block 0 appears more than once, with strict mode enabled."
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

    weights_file_list = glob.glob(str(tmpdir.join("preprocess/*.weights")))
    assert len(weights_file_list) == 0

    # Test oversampling as example weights

    shutil.rmtree(str(tmpdir.join("preprocess")))
    config["data"]["sample_dist"][0]["distribution"] = [
        ["generic", "*3s"],
        ["specific", [5.2, 2]],
        ["news.*pattern", "*2w"],
        [".*something", 1],
    ]

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
    en_file_list = glob.glob(str(tmpdir.join("preprocess/*.en")))
    for ef in en_file_list:
        wf = os.path.splitext(ef)[0] + ".weights"
        assert os.path.isfile(wf)
        if not wf.endswith("news_pattern.weights"):
            with open(wf, "r") as f:
                rf = f.readlines()
                if "corpus_specific" in wf:
                    assert all(el == "2.0\n" for el in rf)
                else:
                    assert all(el == "1.0\n" for el in rf)
                if wf.endswith("generic_corpus.weights"):
                    assert len(rf) == 300
                if wf.endswith("generic_added.weights"):
                    assert len(rf) == 150

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
    en_file_list = glob.glob(str(tmpdir.join("preprocess/*.en")))
    for ef in en_file_list:
        wf = os.path.splitext(ef)[0] + ".weights"
        assert os.path.isfile(wf)
        if not wf.endswith("news_pattern.weights"):
            with open(wf, "r") as f:
                rf = f.readlines()
                if "corpus_specific" in wf:
                    assert all(el == "2.0\n" for el in rf)
                else:
                    assert all(el == "1.0\n" for el in rf)
                if wf.endswith("generic_corpus.weights"):
                    assert len(rf) == 300
                if wf.endswith("generic_added.weights"):
                    assert len(rf) == 150

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


@pytest.mark.parametrize("num_workers", [0, 2])
def test_invalid_unicode_character(tmpdir, num_workers):
    with open(str(tmpdir.join("text.en")), "wb") as f:
        f.write(b"code\n")
    with open(str(tmpdir.join("text.es")), "wb") as f:
        f.write(b"c\xf3digo\n")

    config = {
        "source": "en",
        "target": "es",
        "data": {
            "batch_size": 1,
            "sample": 1,
            "sample_dist": [
                {
                    "path": str(tmpdir),
                    "distribution": [["text", 1]],
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

    preprocessor = TrainingProcessor(config, "", str(tmpdir), num_workers=num_workers)

    with pytest.raises(RuntimeError) as exception_info:
        preprocessor.generate_preprocessed_data()

    error_message = str(exception_info.value)
    assert "text.es" in error_message
    assert "c�digo" in error_message


@prepoperator.register_operator("slow_operator")
class _SlowOperator(prepoperator.Operator):
    def _preprocess(self, tu_batch):
        time.sleep(1)
        return tu_batch


@pytest.mark.parametrize(
    "inactivity_timeout,beat_should_stop", [(None, False), (0.3, True)]
)
@pytest.mark.parametrize("num_workers", [0, 2])
def test_preprocess_activity(tmpdir, inactivity_timeout, beat_should_stop, num_workers):
    corpus_dir = tmpdir.join("corpus")
    corpus_dir.mkdir()
    generate_pseudo_corpus(corpus_dir, 200, "IT", "en")
    generate_pseudo_corpus(corpus_dir, 200, "IT", "de")

    config = {
        "source": "en",
        "target": "de",
        "data": {
            "batch_size": 200,
            "sample": 0,
            "sample_dist": [
                {
                    "path": str(corpus_dir),
                    "distribution": [
                        ["IT", "*"],
                    ],
                }
            ],
        },
        "preprocess": [
            {
                "op": "slow_operator",
            },
            {
                "op": "tokenization",
                "source": {"mode": "space"},
                "target": {"mode": "space"},
            },
        ],
    }

    with requests_mock.Mocker() as m:
        m.register_uri(
            "PUT", "http://test.com/task/beat/1?container_id=abc&duration=0.2"
        )
        beat_service.start_beat_service(
            "abc",
            "http://test.com",
            "1",
            interval=0.1,
            inactivity_timeout=inactivity_timeout,
        )

        preprocessor = TrainingProcessor(
            config, "", str(tmpdir), num_workers=num_workers
        )
        preprocessor.generate_preprocessed_data()

        if beat_should_stop:
            assert not beat_service.beat_service_is_running()
            assert len(m.request_history) == pytest.approx(3, 1)
        else:
            assert beat_service.beat_service_is_running()
            beat_service.stop_beat_service()
            assert not beat_service.beat_service_is_running()
            assert len(m.request_history) == pytest.approx(10, 1)


def test_preprocess_worker_exception(tmpdir):
    @prepoperator.register_operator("exception_operator")
    class _ExceptionOperator(prepoperator.Operator):
        def _preprocess(self, tu_batch):
            raise NotImplementedError()

    corpus_dir = tmpdir.join("corpus")
    corpus_dir.mkdir()
    generate_pseudo_corpus(corpus_dir, 200, "IT", "en")
    generate_pseudo_corpus(corpus_dir, 200, "IT", "de")

    config = {
        "source": "en",
        "target": "de",
        "data": {
            "batch_size": 1,
            "sample": 0,
            "sample_dist": [
                {
                    "path": str(corpus_dir),
                    "distribution": [
                        ["IT", "*"],
                    ],
                }
            ],
        },
        "preprocess": [
            {
                "op": "exception_operator",
            },
        ],
    }

    preprocessor = TrainingProcessor(config, "", str(tmpdir), num_workers=1)
    with pytest.raises(RuntimeError):
        preprocessor.generate_preprocessed_data()


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
    _, vocab_config = preprocessor.generate_vocabularies()
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

    # Test in inference with postprocess
    @prepoperator.register_operator("dummy_op_add_token")
    class DummyOpAddToken(prepoperator.MonolingualOperator):
        def _build_process(self, config, side, build_state):
            return "Not None"

        @property
        def _detok(self):
            return False

        def _apply_process(self, processor, tok):
            tokens = tok.tokens[0]
            tokens[0] = "W"
            return (tok.tokenizer, [tokens])

    config["postprocess"] = [
        {
            "op": "dummy_op_add_token",
        }
    ]

    config["preprocess"].insert(
        0,
        {
            "op": "tokenization",
            "source": {"mode": "char", "joiner_annotate": True},
            "target": {"mode": "char", "joiner_annotate": True},
        },
    )

    prep = InferenceProcessor(config)
    source, target, _ = prep.process_input("This is a test.")
    assert source[0] == ["This", "is", "a", "test", "￭."]  # First and only part.
    assert target[0] is None

    source2, target, _ = prep.process_input("This is a test.", "Das ist...")
    assert source2 == source
    assert target[0] == ["Das", "ist", "￭.", "￭.", "￭."]

    post = InferenceProcessor(config, postprocess=True)
    assert post.process_input(source, target) == "Was ist..."


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
    assert utils.count_lines(output_path) == num_lines


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


@pytest.mark.parametrize("with_metadata", [True, False])
def test_postprocess_multiple_hypotheses(tmpdir, with_metadata):
    src_num_lines = 8
    src_input_path = generate_pseudo_corpus(tmpdir, src_num_lines, "input", "en")
    tgt_num_lines = 8 * 4
    tgt_input_path = generate_pseudo_corpus(tmpdir, tgt_num_lines, "input", "de")
    processor = InferenceProcessor(config_base, postprocess=True)

    metadata = None
    if with_metadata:
        metadata = [[None] for _ in range(src_num_lines)]

    output_path = processor.process_file(src_input_path, tgt_input_path, metadata)

    with open(output_path) as output_file:
        assert len(output_file.readlines()) == tgt_num_lines


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
    assert utils.count_lines(output_path) == 4


def test_postprocess_multipart_file_loader_with_scores(tmpdir):
    src_path = str(tmpdir.join("src.txt"))
    tgt_path = str(tmpdir.join("tgt.txt"))

    with open(src_path, "w") as src_file:
        src_file.write("1 2 3\n")
        src_file.write("4 5\n")
        src_file.write("6\n")
    with open(tgt_path, "w") as tgt_file:
        tgt_file.write("-0.1 ||| a b c\n")
        tgt_file.write("-0.6 ||| d e\n")
        tgt_file.write("-0.2 ||| f\n")

    meta = [[None, None, None]]
    processor = InferenceProcessor(config_base, postprocess=True)
    output_path = processor.process_file(
        src_path,
        tgt_path,
        meta,
        target_score_type=utils.ScoreType.CUMULATED_LL,
    )

    with open(output_path) as output_file:
        assert output_file.read().strip() == "-0.150000 ||| a b c d e f"


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
            return process_type.preprocess

        def _preprocess_tu(self, tu, training):
            tu.add_target("Das ist ein neues Ziel.", "extra")
            assert tu.has_target("extra")
            tu.set_target_output("extra", side="source", delimiter="｟delimiter_token｠")
            return [tu.clone()]

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
            assert "source_lang" in config
            assert "target_lang" in config
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
        config, prepoperator.ProcessType(utils.Task.TRAINING), num_workers=num_workers
    )
    loader = CustomLoader([None, "two", "one", None])
    consumer = CustomConsumer()

    processor.process(loader, consumer)


def test_preprocess_inference_config_with_options():
    @prepoperator.register_operator("politeness")
    class DummyPoliteness(prepoperator.TUOperator):
        def __init__(self, params, process_type, build_state):
            self._default_value = params["default_value"]

        @staticmethod
        def accept_options():
            return True

        def _preprocess_tu(self, tu, meta_batch, options=None):
            value = options.get("value") if options else self._default_value
            politeness_marker = f"｟{value}｠"
            tu.replace_tokens_side("source", (0, 0, [politeness_marker]))
            return [tu]

    config = {
        "source": "en",
        "target": "fr",
        "preprocess": [
            {
                "op": "tokenization",
                "source": {"mode": "aggressive"},
                "target": {"mode": "aggressive"},
            },
            {
                "op": "politeness",
                "name": "politeness-op",
                "default_value": "neutral",
            },
        ],
        "inference_options": {
            "json_schema": {
                "type": "object",
                "properties": {
                    "politeness": {
                        "type": "string",
                        "default": "neutral",
                        "enum": ["formal", "informal", "neutral"],
                    }
                },
            },
            "options": [
                {
                    "option_path": "politeness",
                    "config_path": "preprocess/politeness-op/value",
                },
            ],
        },
    }

    processor = InferenceProcessor(config)
    source, target, _ = processor.process_input("This is a test.")
    assert source == [["｟neutral｠", "This", "is", "a", "test", "."]]

    config["inference"] = {"options": {"politeness": "informal"}}
    processor = InferenceProcessor(config)
    source, target, _ = processor.process_input("This is a test.")
    assert source == [["｟informal｠", "This", "is", "a", "test", "."]]


def test_preprocess_with_inference_config(tmpdir):
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

    processor = InferenceProcessor(config)
    source, target, _ = processor.process_input("2,000", "2,000")
    assert source[0] == ["2", ",", "000"]
    assert target[0] == ["2", ",", "000"]

    config["inference"] = {
        "overrides": {"tokenization_1": {"source": {"mode": "none"}}}
    }

    processor = InferenceProcessor(config)
    source, target, _ = processor.process_input("2,000", "2,000")
    assert source[0] == ["2,000"]
    assert target[0] == ["2", ",", "000"]


def test_inference_preprocess_file_with_target(tmpdir):
    config = {
        "source": "en",
        "target": "de",
        "preprocess": [
            {
                "op": "tokenization",
                "source": {"mode": "aggressive", "joiner_annotate": True},
                "target": {"mode": "aggressive", "joiner_annotate": True},
            },
        ],
    }

    src_path = str(tmpdir.join("src.txt"))
    tgt_path = str(tmpdir.join("tgt.txt"))
    with open(src_path, "w") as src_file:
        src_file.write("Hello world!")
    with open(tgt_path, "w") as tgt_file:
        tgt_file.write("Hallo Welt!")

    processor = InferenceProcessor(config)
    pre_src_path, pre_tgt_path, metadata = processor.process_file(src_path, tgt_path)

    with open(pre_src_path) as pre_src_file:
        assert pre_src_file.read().strip() == "Hello world ￭!"
    with open(pre_tgt_path) as pre_tgt_file:
        assert pre_tgt_file.read().strip() == "Hallo Welt ￭!"


def _check_context_lines(
    context_size,
    random_context,
    res_src_file,
    corpus_src_file,
    res_tgt_file=[],
    corpus_tgt_file=[],
    separator=" ｟mrk_context｠ ",
):
    context = []
    max_context_size = context_size
    context_sizes = set()
    for (
        src_res_line,
        src_corpus_line,
        tgt_res_line,
        tgt_corpus_line,
    ) in itertools.zip_longest(
        res_src_file, corpus_src_file, res_tgt_file, corpus_tgt_file
    ):
        src_res_line = src_res_line.strip()
        src_corpus_line = src_corpus_line.strip()
        if random_context:
            context_size = len(src_res_line.split(separator)) - 1
            context_sizes.add(context_size)
        if tgt_res_line and tgt_corpus_line:
            tgt_res_line = tgt_res_line.strip()
            tgt_corpus_line = tgt_corpus_line.strip()
        if not src_corpus_line or (tgt_corpus_line is not None and not tgt_corpus_line):
            context.clear()
        if context_size:
            src_context_to_print = [
                line[0] for line in context[-context_size:] if line[0].strip()
            ]
            src_line_with_context = separator.join(
                src_context_to_print + [src_corpus_line]
            )
        else:
            src_line_with_context = src_corpus_line
        assert src_res_line == src_line_with_context.strip()
        if tgt_corpus_line:
            if context_size:
                tgt_context_to_print = [
                    line[1] for line in context[-context_size:] if line[1].strip()
                ]
                tgt_line_with_context = separator.join(
                    tgt_context_to_print + [tgt_corpus_line]
                )
            else:
                tgt_line_with_context = tgt_corpus_line
            assert tgt_res_line == tgt_line_with_context.strip()
        if src_corpus_line and (tgt_corpus_line or tgt_corpus_line is None):
            context.append((src_corpus_line, tgt_corpus_line))
            if len(context) > max_context_size:
                del context[0]
    if random_context:
        assert context_sizes == set(range(max_context_size + 1))


@pytest.mark.parametrize(
    "context_size,random_context,no_separator,apply_in_inference,as_main",
    list(
        itertools.product(
            *[
                [1, 2, 3, 5],
                [False, True],
                [False],
                [None, False, True, "split"],
                [False, True],
            ]
        )
    )
    + list(itertools.product(*[[1, 2, 3, 5], [False], [True], [True], [False, True]])),
)
def test_sampler_with_context(
    tmpdir, context_size, random_context, no_separator, apply_in_inference, as_main
):

    random.seed(24)
    corpus_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "corpus", "train"
    )

    config_context = {
        "source": "en",
        "target": "de",
        "data": {
            "sample": 251,
            "sample_dist": [
                {
                    "path": corpus_dir,
                    "distribution": [["europarl", 1, ["context_label"]]],
                }
            ],
            "context": {
                "length": context_size,
                "prob": "random" if random_context else 1,
                "target": True,
                "apply_in_inference": apply_in_inference,
                "as_main": as_main,
                "no_separator": no_separator,
                "labels": ["context_label"],
            },
        },
    }

    separator = " " if no_separator else " ｟mrk_context｠ "
    preprocessor = TrainingProcessor(config_context, "", str(tmpdir))
    (
        data_path,
        train_dir,
        num_samples,
        summary,
        _,
    ) = preprocessor.generate_preprocessed_data()

    with open(
        str(tmpdir.join("preprocess/europarl-v7.de-en.10K.tok.en"))
    ) as res_src_file, open(
        corpus_dir + "/europarl-v7.de-en.10K.tok.en"
    ) as corpus_src_file, open(
        str(tmpdir.join("preprocess/europarl-v7.de-en.10K.tok.de"))
    ) as res_tgt_file, open(
        corpus_dir + "/europarl-v7.de-en.10K.tok.de"
    ) as corpus_tgt_file:
        _check_context_lines(
            context_size,
            random_context,
            res_src_file,
            corpus_src_file,
            res_tgt_file,
            corpus_tgt_file,
            separator=separator,
        )

    # Test file inference.
    src_path = str(tmpdir.join("src.txt"))
    tgt_path = str(tmpdir.join("tgt.txt"))
    with open(src_path, "w") as src_file:
        for line in [
            "Hello world!\n",
            "Nice to meet you.\n",
            "Please be nice.\n",
            " \n",
            "Peace and love.\n",
            "No war.\n",
        ]:
            src_file.write(line)

    with open(tgt_path, "w") as tgt_file:
        for line in [
            "Hallo Welt!\n",
            "Schön dich kennenzulernen.\n",
            "Bitte sei nett.\n",
            "\n",
            "Frieden und Liebe.\n",
            "Kein Krieg.\n",
        ]:
            tgt_file.write(line)

    # Test inference preprocess without target.
    processor = InferenceProcessor(config_context)
    pre_src_path, metadata = processor.process_file(src_path)

    with open(pre_src_path) as pre_src_file, open(src_path) as src_file:
        if apply_in_inference and not no_separator:
            if apply_in_inference == "split":
                context_list = []
                for line in src_file:
                    line = line.strip()
                    if line:
                        if len(context_list) < context_size:
                            context_list.append(line)
                            continue
                        else:
                            line = separator.join(context_list) + separator + line
                    else:
                        if context_list:
                            context_line = separator.join(context_list)
                            assert context_line == pre_src_file.readline().strip()
                    context_list = []
                    assert line == pre_src_file.readline().strip()
                if context_list:
                    assert (
                        separator.join(context_list) == pre_src_file.readline().strip()
                    )
            else:
                _check_context_lines(
                    context_size, False, pre_src_file, src_file, separator=separator
                )
        else:
            for pre_line, line in zip(pre_src_file, src_file):
                assert pre_line.strip() == line.strip()

    # Test postprocess. For simplicity, use source as target.
    postprocessor = InferenceProcessor(config_context, postprocess=True)
    post_src_path = postprocessor.process_file(pre_src_path, pre_src_path)

    if not no_separator:
        with open(src_path) as src_file, open(post_src_path) as post_src_file:
            for source_line, postprocessed_line in zip(src_file, post_src_file):
                assert source_line.strip() == postprocessed_line.strip()

    # Test inference preprocess with target.
    pre_src_path, pre_tgt_path, metadata = processor.process_file(src_path, tgt_path)

    with open(pre_src_path) as pre_src_file, open(src_path) as src_file:
        for pre_l, l in zip(pre_src_file, src_file):
            assert pre_l.strip() == l.strip()
    with open(pre_tgt_path) as pre_tgt_file, open(tgt_path) as tgt_file:
        for pre_l, l in zip(pre_tgt_file, tgt_file):
            assert pre_l.strip() == l.strip()
