# -*- coding: utf-8 -*-

import pytest
import shutil

from nmtwizard.preprocess.preprocess import generate_preprocessed_data
from nmtwizard.preprocess.preprocess import generate_vocabularies
from nmtwizard.preprocess.preprocess import BasicProcessor

def test_sampler(tmpdir):

    corpus_dir = tmpdir.join("corpus")
    corpus_dir.mkdir()

    def _generate_pseudo_corpus(i, name, suffix):
        with corpus_dir.join(name+"."+suffix).open(mode='w') as f :
            for l in range(i):
                f.write(name + " " + str(l) + "\n")

    _generate_pseudo_corpus(800, "corpus_specific1", "en")
    _generate_pseudo_corpus(800, "corpus_specific1", "de")
    _generate_pseudo_corpus(2000, "corpus_specific2", "en")
    _generate_pseudo_corpus(2000, "corpus_specific2", "de")
    _generate_pseudo_corpus(50, "generic_added", "en")
    _generate_pseudo_corpus(50, "generic_added", "de")
    _generate_pseudo_corpus(100, "generic_corpus", "en")
    _generate_pseudo_corpus(100, "generic_corpus", "de")
    _generate_pseudo_corpus(200, "IT", "en")
    _generate_pseudo_corpus(200, "IT", "de")
    _generate_pseudo_corpus(3000, "news_pattern", "en")
    _generate_pseudo_corpus(3000, "news_pattern", "de")

    config = {
        "source": "en",
        "target": "de",
        "data": {
            "sample": 5000,
            "sample_dist": [
                {
                    "path": str(corpus_dir),
                    "distribution": [
                        ["generic", 1],
                        ["specific", 5.2],
                        ["news.*pattern", "*1"],
                        [".*something", 1]
                    ]
                }
            ]
        }
    }

    data_path, train_dir, num_samples, summary, metadata = \
        generate_preprocessed_data(config, "", str(tmpdir))
    assert num_samples == 5000
    assert summary['news_pattern.']['lines_sampled'] == 3000
    assert summary['generic_corpus.']['lines_sampled'] >= 215
    assert summary['generic_added.']['lines_sampled'] >= 107
    assert summary['corpus_specific1.']['lines_sampled'] >= 479
    assert summary['corpus_specific2.']['lines_sampled'] >= 1198
    assert summary['IT.']['lines_sampled'] == 0

    # check unique sampling with undersampling
    with open(str(tmpdir.join("preprocess/corpus_specific2.en")), 'rb') as f :
        rf = f.readlines()
        rf_list = [l.strip() for l in rf]
        assert len(rf_list) == len(set(rf_list))

    # check unique sampling with oversampling
    with open(str(tmpdir.join("preprocess/generic_added.en")), 'rb') as f :
        rf_dict = {}
        for line in f :
            if line in rf_dict:
                rf_dict[line] += 1
            else:
                rf_dict[line] = 1
        for l, c in rf_dict.items() :
            assert 2 <= c <= 3

    shutil.rmtree(str(tmpdir.join("preprocess")))
    config["data"]["sample_dist"] = [
        {
            "path": str(corpus_dir),
            "distribution": [
                ["generic", 1],
                ["specific", 5.2],
                ["news.*pattern", "*2"],
                [".*something", 1]
            ]
        }
    ]
    data_path, train_dir, num_samples, summary, metadata = \
        generate_preprocessed_data(config, "", str(tmpdir))

    assert num_samples == 6000
    assert summary['news_pattern.']['lines_sampled'] == 6000
    assert summary['generic_corpus.']['lines_sampled'] == 0
    assert summary['generic_added.']['lines_sampled'] == 0
    assert summary['corpus_specific1.']['lines_sampled'] == 0
    assert summary['corpus_specific2.']['lines_sampled'] == 0
    assert summary['IT.']['lines_sampled'] == 0

# TODO : test generate vocabularies with several tokenizations
def _test_generate_vocabularies(tmpdir, size, min_frequency, real_size, subword_config=None, multi=False):

    sides = {}
    if multi:
        sides['multi'] = 'en_de'
    else:
        sides['source'] = 'en'
        sides['target'] = 'de'

    config = {
        "source": "en",
        "target": "de",
        "data": {
            "sample": 800,
            "train_dir": ".",
            "sample_dist": [
                {
                    "path": ".",
                    "distribution": [
                        ["europarl", 1]
                    ]
                }
            ]
        },
        "preprocess": [
            {
                "op":"tokenization",
                "source": { "mode": "aggressive" },
                "target": { "mode": "aggressive" },
                "multi": {}
            }
        ]
    }

    for side, ext in sides.items():
        config['preprocess'][0][side]['build_vocabulary'] = {
            "name": "test",
            "size": size,
            "min-frequency": min_frequency,
            "add": ['mama', 'papa'],
            "merge": str(pytest.config.rootdir / "corpus" / "vocab" / "vocab-extra.txt")
        }
        config['preprocess'][0][side]['build_subword'] = subword_config

    _, result_preprocess_config, result_vocab_config  = generate_vocabularies(config, "", str(tmpdir))

    for side, ext in sides.items():

        # Check subword
        if subword_config:
            subword_type = subword_config['type']
            subword_file_name = "subword/"
            if multi:
                subword_file_name += "joint_"
            subword_file_name += "%s_model0-100.%s" % (subword_type, ext)
            subword_file = str(tmpdir.join(subword_file_name))

            if subword_type == 'bpe':
                with open(subword_file, 'rb') as f:
                    assert len(f.readlines()) == 101

            if side == 'multi':
                assert result_preprocess_config[0]['source']['%s_model_path' % subword_type] == subword_file
                assert result_preprocess_config[0]['target']['%s_model_path' % subword_type] == subword_file
            else:
                assert result_preprocess_config[0][side]['%s_model_path' % subword_type] == subword_file

        # Check vocabulary
        rs = real_size[side] if isinstance(real_size, dict) else real_size

        vocab_file_name = "vocabulary/"
        if multi:
            vocab_file_name += "joint_"
        vocab_file_name += "test-%s.%s" % (rs, ext)
        vocab_file = str(tmpdir.join(vocab_file_name))

        with open(vocab_file, 'rb') as f :
            assert len(f.readlines()) == rs

        if side == 'multi':
            assert result_vocab_config['source']['path'] == vocab_file
            assert result_vocab_config['target']['path'] == vocab_file
        else:
            assert result_vocab_config[side]['path'] == vocab_file


def test_generate_vocabularies(tmpdir):

    # Real vocabulary is smaller than size
    _test_generate_vocabularies(tmpdir, 2000, 0, {'source': 1432, 'target': 1704})

    # Real vocabulary is greater than size
    _test_generate_vocabularies(tmpdir, 1000, 0, 1000)

    # Size is greater than filtering by frequency
    _test_generate_vocabularies(tmpdir, 1000, 5, {'source': 636, 'target': 618})

    # Size is smaller than filtering by frequency
    _test_generate_vocabularies(tmpdir, 100, 5, 100)
    
    config_subword_bpe = {
        "params": {
            "vocab_size": 100
        },
        "type": "bpe"
    }

    _test_generate_vocabularies(tmpdir, 50, 5, 50, config_subword_bpe)

    config_subword_sp = {
        "params": {
            "vocab_size": 100
        },
        "type": "sp"
    }

    _test_generate_vocabularies(tmpdir, 50, 5, 50, config_subword_sp)

    _test_generate_vocabularies(tmpdir, 50, 5, 50, config_subword_bpe, True)

def test_preprocess_pipeline(tmpdir):

    config = {
        "source": "en",
        "target": "de",
        "data": {
            "sample": 800,
            "train_dir": ".",
            "sample_dist": [
                {
                    "path": ".",
                    "distribution": [
                        ["europarl", 1]
                    ]
                }
            ]
        },
        "preprocess": [
            {
                "op" : "length_filter",
                "source": {
                    "max_length_char" : 100
                }
            },
            {
                "op" : "tokenization",
                "source": {
                    "mode": "aggressive",
                    "joiner_annotate": True
                },
                "target": {
                    "mode": "aggressive",
                    "joiner_annotate": True
                }
            }
        ]
    }

    data_path, train_dir, num_samples, summary, metadata = \
        generate_preprocessed_data(config, "", str(tmpdir))

    prep = BasicProcessor(config)
    res, _ = prep.process("This is a test.")
    res = res[0]
    assert res == ['This', 'is', 'a', 'test', '￭.']

    prep.build_postprocess_pipeline(config)
    res, _ = prep.process((res, res))
    assert res[0] == "This is a test."
