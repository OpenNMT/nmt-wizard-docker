# -*- coding: utf-8 -*-

import pytest
import shutil
import os
from copy import deepcopy
import random

from nmtwizard.preprocess.preprocess import InferenceProcessor, TrainingProcessor
from nmtwizard.preprocess import prepoperator

def generate_pseudo_corpus(corpus_dir, size, name, suffix):
    with corpus_dir.join(name+"."+suffix).open(mode='w') as f :
        for l in range(size):
            f.write(name + " " + str(l) + "\n")


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
                        ["unaligned", 1]
                    ]
                }
            ]
        }
    }

    preprocessor = TrainingProcessor(config, "", str(tmpdir))
    data_path, train_dir, num_samples, summary, metadata = \
        preprocessor.generate_preprocessed_data()
    assert num_samples == 5000
    assert summary['news_pattern']['linesampled'] == 3000
    assert summary['generic_corpus']['linesampled'] >= 215
    assert summary['generic_added']['linesampled'] >= 107
    assert summary['corpus_specific1']['linesampled'] >= 479
    assert summary['corpus_specific2']['linesampled'] >= 1198
    assert summary['IT']['linesampled'] == 0
    assert 'unaligned' not in summary
    assert summary['generic_to_ignore']['linesampled'] == 0

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

    # Check strict mode
    shutil.rmtree(str(tmpdir.join("preprocess")))
    config["data"]["mode_strict"] = True

    preprocessor = TrainingProcessor(config, "", str(tmpdir))
    with pytest.raises(RuntimeError) as excinfo:
        data_path, train_dir, num_samples, summary, metadata = \
            preprocessor.generate_preprocessed_data()
    assert str(excinfo.value) == "pattern '.*something' in block 0 doesn't match any file with strict mode enabled."

    shutil.rmtree(str(tmpdir.join("preprocess")))
    config["data"].pop("mode_strict")
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

    preprocessor = TrainingProcessor(config, "", str(tmpdir))
    data_path, train_dir, num_samples, summary, metadata = \
        preprocessor.generate_preprocessed_data()

    assert num_samples == 6000
    assert summary['news_pattern']['linesampled'] == 6000
    assert summary['generic_corpus']['linesampled'] == 0
    assert summary['generic_added']['linesampled'] == 0
    assert summary['corpus_specific1']['linesampled'] == 0
    assert summary['corpus_specific2']['linesampled'] == 0
    assert summary['IT']['linesampled'] == 0

    del os.environ["NB_CPU"]


def test_sampler_with_annotations(tmpdir):

    with open(str(tmpdir.join("train.en")), "w") as en:
        en.write("\n".join(["1", "2", "3", "4", "5", "6"]))
    with open(str(tmpdir.join("train.fr")), "w") as fr:
        fr.write("\n".join(["1", "2", "3", "4", "5", "6"]))

    annot_dir = tmpdir.join("train_enfr_annot")
    os.makedirs(str(annot_dir))
    with open(str(annot_dir.join("train")), "w") as annot:
        annot.write("\n".join(["0.0274","-0.1201", "0.2499", "0.8566", "-0.8025", "0.0892"]))

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
                    "distribution" : [ ["train", "*"] ],
                    "annotations":{
                        "similarity": "train_enfr_annot"
                    }
                }
            ]
        }
    }

    preprocessor = TrainingProcessor(config, from_dir, to_dir)
    data_path, train_dir, num_samples, summary, metadata = preprocessor.generate_preprocessed_data()

    assert 'annotations' in summary['train'] and summary['train']['annotations'] == ['similarity']


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
            "sample_dist": [
                {
                    "path": os.path.join(os.path.dirname(os.path.realpath(__file__)), "corpus", "train"),
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
            "merge": os.path.join(os.path.dirname(os.path.realpath(__file__)), "corpus", "vocab", "vocab-extra.txt")
        }
        config['preprocess'][0][side]['build_subword'] = subword_config

    preprocessor = TrainingProcessor(config, "", str(tmpdir))
    _, result_preprocess_config, result_vocab_config  = preprocessor.generate_vocabularies()

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
                        ["generic", 1, "generic_tag"],
                        ["extra_generic", 1, "generic_tag"],
                        ["good_news", 1, "news_tag"],
                        ["news", 1, "news_tag"],
                        ["no_tag", 1]
                    ]
                }
            ]
        },
        "preprocess": [
            {
                "op" : "length_filter",
                "source": {
                    "max_length_char" : 3
                },
                "overrides": {
                    "generic_tag": {
                        "disabled": True
                    },
                    "news_tag": {
                        "source": {
                            "max_length_char" : 6
                        }
                    }
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

    generate_pseudo_corpus(corpus_dir, 10, "generic", "en")
    generate_pseudo_corpus(corpus_dir, 10, "generic", "de")
    generate_pseudo_corpus(corpus_dir, 10, "extra_generic", "en")
    generate_pseudo_corpus(corpus_dir, 10, "extra_generic", "de")
    generate_pseudo_corpus(corpus_dir, 10, "good_news", "en")
    generate_pseudo_corpus(corpus_dir, 10, "good_news", "de")
    generate_pseudo_corpus(corpus_dir, 10, "news", "en")
    generate_pseudo_corpus(corpus_dir, 10, "news", "de")
    generate_pseudo_corpus(corpus_dir, 10, "no_tag", "en")
    generate_pseudo_corpus(corpus_dir, 10, "no_tag", "de")


    preprocessor = TrainingProcessor(config, "", str(tmpdir))
    data_path, train_dir, num_samples, summary, metadata = \
        preprocessor.generate_preprocessed_data()

    for f_name, f_info in summary.items():
        if "generic" in f_name or f_name == 'news':
            # generic is not filtered because filtering is disabled
            # 'news' is not filtered because length is overriden
            assert f_info['linefiltered'] == f_info['linesampled']
        else :
            assert f_info['linefiltered'] == 0

    prep = InferenceProcessor(config)
    source, target = prep.process_input("This is a test.")
    source_no_meta = source[0]
    assert source_no_meta[0] == ['This', 'is', 'a', 'test', '￭.'] # First and only part.
    assert target == [None]

    source2, target = prep.process_input(("This is a test.", "Das ist..."))
    assert source2 == source
    assert target[0] == ['Das', 'ist', '￭.', '￭.', '￭.']

    post = InferenceProcessor(config, postprocess=True)
    target_postprocessed = post.process_input((source, target))
    assert target_postprocessed == "Das ist..."


config_base = {
    "source": "en",
    "target": "de",
    "data": {
        "sample": 800,
        "sample_dist": [
            {
                "path": os.path.join(os.path.dirname(os.path.realpath(__file__)), "corpus", "train"),
                "distribution": [
                    ["europarl", 1]
                ]
            }
        ]
    },
    "preprocess": [
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
        },
        {
            "op": "alignment",
            "write_alignment": True,
            "forward": {
                "probs": os.path.join(os.path.dirname(os.path.realpath(__file__)), "corpus", "resources", "alignment", "ende_forward.probs")
            },
            "backward": {
                "probs": os.path.join(os.path.dirname(os.path.realpath(__file__)), "corpus", "resources", "alignment", "ende_backward.probs")
            }
        }
    ]
}

def test_preprocess_align(tmpdir):

    preprocessor = TrainingProcessor(config_base, "", str(tmpdir))
    data_path, train_dir, num_samples, summary, metadata = \
        preprocessor.generate_preprocessed_data()

    with open(os.path.join(str(tmpdir), "preprocess", "europarl-v7.de-en.10K.tok.align")) as align:
        assert align.readline().strip() == "0-0 1-0 2-1 3-2"


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

                src_num_to_del = random.randint(0, src_len-src_pos)
                tgt_num_to_del = random.randint(0, tgt_len-tgt_pos)

                src_tok_replace = replacement_tokens[0:random.randint(0, len(replacement_tokens))]
                tgt_tok_replace = replacement_tokens[0:random.randint(0, len(replacement_tokens))]
                yield ((src_pos, src_num_to_del, src_tok_replace), (tgt_pos, tgt_num_to_del, tgt_tok_replace))


        def _preprocess_tu(self, tu, training):

            def joiner_side(tokens, pos, num_to_del):
                joiner_start = False
                joiner_end = False
                if num_to_del:
                    joiner_start = tokens[pos].startswith(joiner_marker)
                    joiner_end = tokens[pos+num_to_del-1].endswith(joiner_marker)
                return joiner_start, joiner_end

            def checks_side(tokens, length, pos, num_to_del, tok_replace, joiner_start, joiner_end):
                assert(len(tokens) == length - num_to_del + len(tok_replace))

                if tok_replace:
                    if joiner_start:
                        tok_replace[0] = joiner_marker + tok_replace[0]
                    if joiner_end:
                        tok_replace[-1] += joiner_marker
                    assert (tokens[pos:pos+len(tok_replace)] == tok_replace)

            def change_align_side(al_idx, pos, num_to_del, tok_replace):
                new_al_idx = al_idx
                len_tok_replace = len(tok_replace) if tok_replace else 0
                if al_idx >= pos:
                    if al_idx < pos+num_to_del:
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
                alignment_before = deepcopy(tu.alignment)

                joiner_marker = "￭"

                next(self._rand_repl_gen)
                src_len = len(tu.src_tok.tokens)
                tgt_len = len(tu.tgt_tok.tokens)
                src_replace, tgt_replace = self._rand_repl_gen.send((src_len, tgt_len))

                src_pos, src_num_to_del, src_tok_replace = src_replace
                tgt_pos, tgt_num_to_del, tgt_tok_replace = tgt_replace

                src_joiner_start, src_joiner_end = joiner_side(
                    tu.src_tok.tokens, src_pos, src_num_to_del)
                tgt_joiner_start, tgt_joiner_end = joiner_side(
                    tu.tgt_tok.tokens, tgt_pos, tgt_num_to_del)

                tu.replace_tokens(src_replace, tgt_replace)

                checks_side(
                    tu.src_tok.tokens,
                    src_len,
                    src_pos,
                    src_num_to_del,
                    src_tok_replace,
                    src_joiner_start,
                    src_joiner_end)
                checks_side(
                    tu.tgt_tok.tokens,
                    tgt_len,
                    tgt_pos,
                    tgt_num_to_del,
                    tgt_tok_replace,
                    tgt_joiner_start,
                    tgt_joiner_end)

                # Check alignment
                alignment_after = tu.alignment

                new_align = set()
                for al_src, al_tgt in alignment_before :
                    new_al_src = change_align_side(al_src, src_pos, src_num_to_del, src_tok_replace)
                    if new_al_src is None:
                        continue
                    new_al_tgt = change_align_side(al_tgt, tgt_pos, tgt_num_to_del, tgt_tok_replace)
                    if new_al_tgt is None:
                        continue
                    new_align.add((new_al_src, new_al_tgt))

                assert new_align == alignment_after
            return [tu]

    config_replace = deepcopy(config_base)

    config_replace['preprocess'].append(
        { "op": "repl" }
    )

    preprocessor = TrainingProcessor(config_replace, "", str(tmpdir))
    data_path, train_dir, num_samples, summary, metadata = \
        preprocessor.generate_preprocessed_data()
