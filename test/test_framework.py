# -*- coding: utf-8 -*-

import pytest
import os
import tempfile
import shutil
import json
import copy
import filecmp
import six
import functools

from nmtwizard.framework import Framework
from nmtwizard.preprocess import preprocess
from nmtwizard.preprocess import prepoperator
from nmtwizard.serving import TranslationOutput, run_request


class DummyCheckpoint(object):
    """Dummy checkpoint files for testing."""

    def __init__(self, model_dir):
        self._model_dir = model_dir
        self._files = [
            os.path.join(model_dir, "checkpoint.bin"),
            os.path.join(model_dir, "metadata.txt"),
        ]

    def exists(self):
        return all(os.path.isfile(f) for f in self._files)

    def corrupt(self):
        with open(self._files[0], "w") as model_file:
            model_file.write("invalid data")

    def index(self):
        assert self.exists()
        index = None
        for path in self._files:
            with open(path, "r") as model_file:
                content = model_file.read().split(" ")
                assert content[0] == os.path.basename(path)
                read_index = int(content[1])
                if index is None:
                    index = read_index
                else:
                    assert read_index == index
        return index

    def objects(self):
        assert self.exists()
        return {os.path.basename(path): path for path in self._files}

    def build(self, index):
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)
        for path in self._files:
            key = os.path.basename(path)
            with open(path, "w") as model_file:
                content = "%s %d" % (key, index)
                model_file.write(content)
        return self.objects()


class _TestFramework(Framework):
    def _map_vocab_entry(self, index, token, converted_vocab):
        converted_vocab.write(token)
        converted_vocab.write("\n")

    def train(self, *args, **kwargs):
        pass

    def trans(self, *args, **kwargs):
        pass

    def release(self, *args, **kwargs):
        pass

    def serve(self, *args, **kwargs):
        pass

    def forward_request(self, *args, **kwargs):
        pass


class DummyFramework(_TestFramework):
    """Dummy framework for testing."""

    def train(
        self,
        config,
        src_file,
        tgt_file,
        src_vocab_info,
        tgt_vocab_info,
        align_file=None,
        example_weights_file=None,
        model_path=None,
        gpuid=0,
    ):
        # Verify that input files exist.
        assert os.path.exists(src_file)
        assert os.path.exists(tgt_file)
        assert "source" in config["vocabulary"] and os.path.exists(
            config["vocabulary"]["source"]["path"]
        )

        assert "target" in config["vocabulary"] and os.path.exists(
            config["vocabulary"]["target"]["path"]
        )

        # Generate some checkpoint files.
        index = 0
        if model_path is not None:
            parent_ckpt = DummyCheckpoint(model_path)
            if parent_ckpt.exists():
                index = parent_ckpt.index() + 1
        model_dir = os.path.join(self._output_dir, "model")
        return DummyCheckpoint(model_dir).build(index)

    def trans(self, config, model_path, input_path, output_path, gpuid=None):
        # Reverse the input.
        with open(input_path) as input_file, open(output_path, "w") as output_file:
            for line in input_file:
                output_file.write(" ".join(reversed(line.split())))

    def release(self, config, model_path, optimization_level=None, gpuid=0):
        return DummyCheckpoint(model_path).objects()

    def _translate(self, source, target):
        if target is None:
            target = []
        target += list(reversed(source))
        return target

    def serve(self, config, model_path, gpuid=0):
        return None, self._translate

    def forward_request(self, model_info, inputs, outputs=None, options=None):
        translate_fn = model_info
        hypotheses = []
        for i, source in enumerate(inputs):
            target = outputs[i] if outputs is not None else None
            output = translate_fn(source, target)
            hypotheses.append([TranslationOutput(output)])
        return hypotheses


def _get_lines(path):
    with open(path, "r") as f:
        lines = []
        for line in f:
            lines.append(line.strip())
        return lines


def _check_vocab(current, previous, changed=False, added_tokens=None, vocab_size=None):
    assert os.path.exists(current)
    if changed or added_tokens:
        assert previous is not None
        assert os.path.exists(previous)
        assert not filecmp.cmp(current, previous, shallow=False)
    else:
        assert previous is None
    curr_tokens = _get_lines(current)
    if vocab_size is not None:
        assert len(curr_tokens) == vocab_size
    if added_tokens:
        if vocab_size is None:
            prev_tokens = _get_lines(previous)
            assert len(curr_tokens) == len(prev_tokens) + len(added_tokens)
        curr_vocab = set(curr_tokens)
        for token in added_tokens:
            assert token in curr_vocab


class ReplaceVocabChecker(_TestFramework):
    def __init__(
        self,
        src_changed=False,
        tgt_changed=False,
        src_tokens_to_add=None,
        tgt_tokens_to_add=None,
        src_vocab_size=None,
        tgt_vocab_size=None,
        joint=False,
    ):
        super(ReplaceVocabChecker, self).__init__()
        self.src_changed = src_changed
        self.tgt_changed = tgt_changed
        self.src_tokens_to_add = src_tokens_to_add or []
        self.tgt_tokens_to_add = tgt_tokens_to_add or []
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.joint = joint

    def train(
        self,
        config,
        src_file,
        tgt_file,
        src_vocab_info,
        tgt_vocab_info,
        align_file=None,
        example_weights_file=None,
        model_path=None,
        gpuid=0,
    ):
        if self.joint:
            assert src_vocab_info.current == tgt_vocab_info.current
            assert src_vocab_info.previous == tgt_vocab_info.previous
            changed = self.src_changed or self.tgt_changed
            tokens_to_add = set(self.src_tokens_to_add + self.tgt_tokens_to_add)
            vocab_size = self.src_vocab_size
            _check_vocab(
                src_vocab_info.current,
                src_vocab_info.previous,
                changed,
                tokens_to_add,
                vocab_size,
            )
            _check_vocab(
                tgt_vocab_info.current,
                tgt_vocab_info.previous,
                changed,
                tokens_to_add,
                vocab_size,
            )
        else:
            _check_vocab(
                src_vocab_info.current,
                src_vocab_info.previous,
                self.src_changed,
                self.src_tokens_to_add,
                self.src_vocab_size,
            )
            _check_vocab(
                tgt_vocab_info.current,
                tgt_vocab_info.previous,
                self.tgt_changed,
                self.tgt_tokens_to_add,
                self.tgt_vocab_size,
            )
        model_dir = os.path.join(self._output_dir, "model")
        os.makedirs(model_dir)
        checkpoint_path = os.path.join(model_dir, "checkpoint.txt")
        with open(checkpoint_path, "w") as checkpoint:
            checkpoint.write("5")
        return {os.path.basename(checkpoint_path): checkpoint_path}

    def _generate_training_data(self, config):
        outputs = list(super(ReplaceVocabChecker, self)._generate_training_data(config))
        outputs[-1] = dict(source=self.src_tokens_to_add, target=self.tgt_tokens_to_add)
        return tuple(outputs)


config_base = {
    "source": "en",
    "target": "de",
    "preprocess": [
        {
            "op": "tokenization",
            "source": {"mode": "aggressive", "joiner_annotate": True},
            "target": {"mode": "aggressive", "joiner_annotate": True},
        }
    ],
    "vocabulary": {
        "source": {"path": "${CORPUS_DIR}/vocab/en-vocab.txt"},
        "target": {"path": "${CORPUS_DIR}/vocab/de-vocab.txt"},
    },
    "options": {},
}

config_base_old = {
    "source": "en",
    "target": "de",
    "tokenization": {
        "source": {
            "vocabulary": "${CORPUS_DIR}/vocab/en-vocab.txt",
            "mode": "aggressive",
            "joiner_annotate": True,
        },
        "target": {
            "vocabulary": "${CORPUS_DIR}/vocab/de-vocab.txt",
            "mode": "aggressive",
            "joiner_annotate": True,
        },
    },
    "options": {},
}


def _clear_workspace(tmpdir):
    tmpdir = str(tmpdir)
    workspace_dir = os.path.join(tmpdir, "workspace")
    shutil.rmtree(workspace_dir)


def _read_config(model_dir):
    config_path = os.path.join(model_dir, "config.json")
    assert os.path.isfile(config_path)
    with open(config_path) as config_file:
        return json.load(config_file)


def _test_dir():
    return os.path.dirname(os.path.realpath(__file__))


def _run_framework(
    tmpdir,
    task_id,
    args,
    config=None,
    parent=None,
    auto_ms=True,
    corpus_env=True,
    env=None,
    storage_config=None,
    framework_fn=None,
):
    base_environ = os.environ.copy()
    os.environ["MODELS_DIR"] = str(tmpdir.join("models"))
    os.environ["WORKSPACE_DIR"] = str(tmpdir.join("workspace"))
    if corpus_env:
        os.environ["CORPUS_DIR"] = os.path.join(_test_dir(), "corpus")
    if env is not None:
        os.environ.update(env)
    full_args = ["-t", str(task_id)]
    if storage_config is not None:
        full_args += ["-s", json.dumps(storage_config)]
    if auto_ms:
        full_args += ["-ms", os.environ["MODELS_DIR"]]
    if config is not None:
        full_args += ["-c", json.dumps(config)]
    if parent is not None:
        full_args += ["-m", parent]
    if isinstance(args, six.string_types):
        args = args.split(" ")
    full_args += args
    if framework_fn is None:
        framework = DummyFramework()
    else:
        framework = framework_fn()
    framework.run(args=full_args)
    _clear_workspace(tmpdir)
    model_dir = os.path.join(os.environ["MODELS_DIR"], task_id)
    os.environ.clear()
    os.environ.update(base_environ)
    return model_dir


def _check_dir(directory, files):
    assert sorted(os.listdir(directory)) == sorted(files)


def test_train(tmpdir):
    model_dir = _run_framework(tmpdir, "model0", "train", config=config_base)
    config = _read_config(model_dir)
    assert config["model"] == "model0"
    assert config["modelType"] == "checkpoint"
    assert os.path.isfile(
        os.path.join(
            model_dir, os.path.basename(config["vocabulary"]["source"]["path"])
        )
    )
    assert os.path.isfile(
        os.path.join(
            model_dir, os.path.basename(config["vocabulary"]["target"]["path"])
        )
    )

    model_dir_old = _run_framework(tmpdir, "model1", "train", config=config_base_old)
    config = _read_config(model_dir_old)
    assert os.path.isfile(
        os.path.join(
            model_dir_old,
            os.path.basename(config["tokenization"]["source"]["vocabulary"]),
        )
    )
    assert os.path.isfile(
        os.path.join(
            model_dir_old,
            os.path.basename(config["tokenization"]["target"]["vocabulary"]),
        )
    )
    assert "vocabulary" not in config
    assert "preprocess" not in config

    assert DummyCheckpoint(model_dir).index() == 0


def test_train_with_storage_in_config(tmpdir):
    storage_config = {
        "corpus": {"type": "local", "basedir": os.path.join(_test_dir(), "corpus")}
    }
    config = {
        "source": "en",
        "target": "de",
        "preprocess": [
            {
                "op": "tokenization",
                "source": {"mode": "aggressive", "joiner_annotate": True},
                "target": {"mode": "aggressive", "joiner_annotate": True},
            }
        ],
        "vocabulary": {
            "source": {"path": "${LOCAL_DIR}/vocab/en-vocab.txt"},
            "target": {"path": "${LOCAL_DIR}/vocab/de-vocab.txt"},
        },
        "options": {},
    }
    model_dir = _run_framework(
        tmpdir,
        "model0",
        "train",
        config=config,
        env={"LOCAL_DIR": "corpus:"},
        storage_config=storage_config,
    )
    config = _read_config(model_dir)
    assert config["vocabulary"]["source"]["path"] == "${MODEL_DIR}/en-vocab.txt"
    assert config["vocabulary"]["target"]["path"] == "${MODEL_DIR}/de-vocab.txt"
    assert os.path.isfile(os.path.join(model_dir, "de-vocab.txt"))
    assert os.path.isfile(os.path.join(model_dir, "en-vocab.txt"))


def test_train_with_sampling(tmpdir):
    def _make_sampling_config(n):
        config = copy.deepcopy(config_base)
        config["data"] = {
            "sample": n,
            "train_dir": ".",
            "sample_dist": [{"path": ".", "distribution": [["europarl", 1]]}],
        }
        return config

    model_dir = _run_framework(
        tmpdir, "model0", "train", config=_make_sampling_config(1000)
    )
    config = _read_config(model_dir)
    assert config["build"]["sentenceCount"] == 1000
    assert config["build"]["cumSentenceCount"] == 1000
    model_dir = _run_framework(
        tmpdir, "model1", "train", config=_make_sampling_config(800), parent="model0"
    )
    config = _read_config(model_dir)
    assert config["build"]["sentenceCount"] == 800
    assert config["build"]["cumSentenceCount"] == 1800


def test_train_with_sampling_v2(tmpdir):
    config = {
        "source": "en",
        "target": "de",
        "data": {
            "sample": 1000,
            "sample_dist": [
                {"path": "${DATA_DIR}/train", "distribution": [["europarl", 1]]}
            ],
        },
        "preprocess": [
            {
                "op": "tokenization",
                "source": {"mode": "aggressive", "joiner_annotate": True},
                "target": {"mode": "aggressive", "joiner_annotate": True},
            }
        ],
        "vocabulary": {
            "source": {"path": "${DATA_DIR}/vocab/en-vocab.txt"},
            "target": {"path": "${DATA_TRAIN_DIR}/vocab/de-vocab.txt"},
        },
        "options": {},
    }
    os.environ["DATA_DIR"] = os.path.join(_test_dir(), "corpus")
    model_dir = _run_framework(
        tmpdir, "model0", "train", config=config, corpus_env=False
    )
    config = _read_config(model_dir)
    assert config["build"]["sentenceCount"] == 1000
    assert os.path.exists(os.path.join(model_dir, "en-vocab.txt"))
    assert not os.path.exists(os.path.join(model_dir, "de-vocab.txt"))
    assert not os.path.exists(os.path.join(model_dir, "train"))
    del os.environ["DATA_DIR"]


def test_train_chain(tmpdir):
    _run_framework(tmpdir, "model0", "train", config=config_base)
    model_dir = _run_framework(tmpdir, "model1", "train", parent="model0")
    config = _read_config(model_dir)
    assert config["parent_model"] == "model0"
    assert config["model"] == "model1"
    assert config["modelType"] == "checkpoint"
    assert DummyCheckpoint(model_dir).index() == 1


@pytest.mark.parametrize(
    "config,custom_field,new_field,mode,expected_field",
    [
        (config_base_old, {"a": 1, "b": 2}, {"a": 3}, "default", {"a": 3, "b": 2}),
        (config_base_old, {"a": 1, "b": 2}, {"a": 3}, "replace", {"a": 3}),
    ],
)
def test_config_update(tmpdir, config, custom_field, new_field, mode, expected_field):
    config["custom_field"] = custom_field
    _run_framework(tmpdir, "model0", "train", config=config)
    new_config = {"custom_field": {"a": 3}}
    model_dir = _run_framework(
        tmpdir,
        "model1",
        ["--config_update_mode", mode, "train"],
        parent="model0",
        config=new_config,
    )
    config = _read_config(model_dir)
    assert config["custom_field"] == expected_field


def test_config_v2_upgrade(tmpdir):
    _run_framework(tmpdir, "model0", "train", config=config_base_old)

    class _CheckV2ConfigFramework(DummyFramework):
        def _get_preprocessor(self, config, train=True):
            # The configuration should be fully upgraded to V2.
            assert "tokenization" not in config
            assert "preprocess" in config
            return super()._get_preprocessor(config, train=train)

    model_dir = _run_framework(
        tmpdir,
        "model1",
        "train",
        config=config_base,
        parent="model0",
        framework_fn=_CheckV2ConfigFramework,
    )

    config = _read_config(model_dir)
    assert "tokenization" not in config
    assert "preprocess" in config


def test_model_storage(tmpdir):
    ms1 = tmpdir.join("ms1")
    ms2 = tmpdir.join("ms2")
    _run_framework(
        tmpdir, "model0", ["-ms", str(ms1), "train"], config=config_base, auto_ms=False
    )
    assert ms1.join("model0").check(dir=1, exists=1)
    _run_framework(
        tmpdir,
        "model1",
        ["-msr", str(ms1), "-msw", str(ms2), "train"],
        parent="model0",
        auto_ms=False,
    )
    assert ms1.join("model1").check(exists=0)
    assert ms2.join("model1").check(dir=1, exists=1)


def test_release(tmpdir):
    _run_framework(tmpdir, "model0", "train", config=config_base)
    _run_framework(tmpdir, "model1", "train", parent="model0")
    _run_framework(tmpdir, "release0", "release", parent="model1")
    model_dir = str(tmpdir.join("models").join("model1_release"))
    config = _read_config(model_dir)
    assert "parent_model" not in config
    assert "build" not in config
    assert "data" not in config
    assert config["model"] == "model1_release"
    assert config["modelType"] == "release"
    assert DummyCheckpoint(model_dir).index() == 1
    assert os.path.isfile(
        os.path.join(
            model_dir, os.path.basename(config["vocabulary"]["source"]["path"])
        )
    )
    assert os.path.isfile(
        os.path.join(
            model_dir, os.path.basename(config["vocabulary"]["target"]["path"])
        )
    )


def test_release_change_file(tmpdir):
    _run_framework(tmpdir, "model0", "train", config=config_base)

    new_vocab = "vocab.src"
    with open(str(tmpdir.join(new_vocab)), "w") as vocab_src:
        vocab_src.write("0\n")
    override = {"vocabulary": {"source": {"path": "${TMP_DIR}/%s" % new_vocab}}}
    _run_framework(
        tmpdir,
        "release0",
        "release",
        parent="model0",
        config=override,
        env={"TMP_DIR": str(tmpdir)},
    )
    model_dir = str(tmpdir.join("models").join("model0_release"))
    config = _read_config(model_dir)
    assert config["vocabulary"]["source"]["path"] == "${MODEL_DIR}/%s" % new_vocab
    assert os.path.isfile(
        os.path.join(
            model_dir, os.path.basename(config["vocabulary"]["source"]["path"])
        )
    )


# Dummy domain classifier operator.
@prepoperator.register_operator("domain")
class _DomainClassifier(prepoperator.Operator):
    @classmethod
    def _config_schema(cls):
        schema = super(_DomainClassifier, cls)._config_schema()

        schema["properties"].update(
            {"source": {"type": "object"}, "target": {"type": "object"}}
        )
        return schema

    def _preprocess(self, tu_batch):
        return tu_batch


def test_release_with_inference_options(tmpdir):
    config = copy.deepcopy(config_base)
    config["preprocess"].append({"op": "domain", "source": {"some_training_field": {}}})
    # TODO V2 : Deal with inference options
    config["inference_options"] = {
        "json_schema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "title": "Domain", "enum": ["IT", "News"]}
            },
        },
        "options": [{"option_path": "domain", "config_path": "preprocess/0/source"}],
    }
    config["supported_features"] = {"my_feature": True}
    _run_framework(tmpdir, "model0", "train", config=config)
    _run_framework(tmpdir, "release0", "release", parent="model0")
    model_dir = str(tmpdir.join("models").join("model0_release"))
    options_path = os.path.join(model_dir, "options.json")
    assert os.path.exists(options_path)
    with open(options_path) as options_file:
        model_options = json.load(options_file)
        assert (
            model_options["json_schema"] == config["inference_options"]["json_schema"]
        )
        assert model_options["supported_features"] == config["supported_features"]


def test_integrity_check(tmpdir):
    model_dir = _run_framework(tmpdir, "model0", "train", config=config_base)
    DummyCheckpoint(model_dir).corrupt()
    with pytest.raises(RuntimeError):
        _run_framework(tmpdir, "model1", "train", parent="model0")


def test_preprocess_as_model(tmpdir):
    model_dir = _run_framework(
        tmpdir, "preprocess0", "preprocess --build_model", config=config_base
    )
    config = _read_config(model_dir)
    assert config["model"] == "preprocess0"
    assert config["modelType"] == "preprocess"
    assert os.path.isfile(
        os.path.join(model_dir, "data", "train.%s" % config["source"])
    )
    assert os.path.isfile(
        os.path.join(model_dir, "data", "train.%s" % config["target"])
    )


def test_preprocess_sample_with_output(tmpdir):
    def verify_output_file(filename):
        outputfile = os.path.join(_test_dir(), "corpus", filename)
        with open(outputfile) as file:
            assert len(file.readlines()) == 1000
        os.remove(outputfile)

    storage_config = {
        "corpus": {"type": "local", "basedir": os.path.join(_test_dir(), "corpus")}
    }
    config = {
        "source": "en",
        "data": {
            "sample": 1000,
            "sample_dist": [
                {"path": "${DATA_DIR}/train", "distribution": [["europarl", 1]]}
            ],
        },
    }
    os.environ["DATA_DIR"] = os.path.join(_test_dir(), "corpus")
    _run_framework(
        tmpdir,
        "preprocess0",
        "preprocess -o corpus:sample",
        config=config,
        storage_config=storage_config,
    )
    verify_output_file("sample.en")

    # sample bitext
    config["target"] = "de"
    _run_framework(
        tmpdir,
        "preprocess0",
        "preprocess -o corpus:sample",
        config=config,
        storage_config=storage_config,
    )
    verify_output_file("sample.en")
    verify_output_file("sample.de")


def test_description(tmpdir):
    description = "A description with a non ascii character: é"
    config = copy.deepcopy(config_base)
    config["description"] = description
    model_dir = _run_framework(tmpdir, "train", "train", config=config)
    readme = os.path.join(model_dir, "README.md")
    assert os.path.exists(readme)
    with open(readme) as readme_file:
        assert readme_file.read().strip() == description


def test_preprocess_train_chain(tmpdir):
    _run_framework(
        tmpdir, "preprocess0", "preprocess --build_model", config=config_base
    )
    model_dir = _run_framework(tmpdir, "model0", "train", parent="preprocess0")
    config = _read_config(model_dir)
    assert "parent_model" not in config
    assert config["model"] == "model0"
    assert config["modelType"] == "checkpoint"
    assert not os.path.isdir(os.path.join(model_dir, "data"))
    model_dir = _run_framework(
        tmpdir, "preprocess1", "preprocess --build_model", parent="model0"
    )
    config = _read_config(model_dir)
    assert config["model"] == "preprocess1"
    assert config["modelType"] == "preprocess"
    assert os.path.isdir(os.path.join(model_dir, "data"))
    assert DummyCheckpoint(model_dir).index() == 0  # The checkpoints were forwarded.
    model_dir = _run_framework(tmpdir, "model1", "train", parent="preprocess1")
    config = _read_config(model_dir)
    assert config["parent_model"] == "model0"  # The parent is the previous training.
    assert config["model"] == "model1"
    assert config["modelType"] == "checkpoint"
    assert not os.path.isdir(os.path.join(model_dir, "data"))
    assert DummyCheckpoint(model_dir).index() == 1


def _test_buildvocab(tmpdir, run_num, multi=False):

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
            "sample": 1000,
            "sample_dist": [
                {"path": "${DATA_DIR}/train", "distribution": [["europarl", 1]]}
            ],
        },
        "options": {},
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
            "size": 50,
            "min-frequency": 5,
        }

    os.environ["DATA_DIR"] = os.path.join(_test_dir(), "corpus")

    model_dir = _run_framework(tmpdir, "buildvocab%d" % run_num, "buildvocab", config)
    config_buildvocab = _read_config(model_dir)
    assert "parent_model" not in config_buildvocab
    assert config_buildvocab["model"] == "buildvocab%d" % run_num
    assert config_buildvocab["modelType"] == "base"
    assert not os.path.isdir(os.path.join(model_dir, "data"))

    model_dir = _run_framework(
        tmpdir, "model%d" % run_num, "train", parent="buildvocab%d" % run_num
    )
    config_checkpoint = _read_config(model_dir)
    assert config_checkpoint["model"] == "model%d" % run_num
    assert config_checkpoint["parent_model"] == "buildvocab%d" % run_num
    assert config_checkpoint["modelType"] == "checkpoint"
    assert not os.path.isdir(os.path.join(model_dir, "data"))

    for side, ext in sides.items():
        vocab_file_name = "${MODEL_DIR}/"
        if multi:
            vocab_file_name += "joint_"
        vocab_file_name += "vocab_test-50.%s" % ext

        if side == "multi":
            assert config_buildvocab["vocabulary"]["source"]["path"] == vocab_file_name
            assert config_checkpoint["vocabulary"]["source"]["path"] == vocab_file_name

            assert config_buildvocab["vocabulary"]["target"]["path"] == vocab_file_name
            assert config_checkpoint["vocabulary"]["target"]["path"] == vocab_file_name
        else:
            assert config_buildvocab["vocabulary"][side]["path"] == vocab_file_name
            assert config_checkpoint["vocabulary"][side]["path"] == vocab_file_name

    del os.environ["DATA_DIR"]


def test_buildvocab(tmpdir):
    _test_buildvocab(tmpdir, 0)
    _test_buildvocab(tmpdir, 1, True)


def test_replace_vocab_old(tmpdir):
    _run_framework(
        tmpdir,
        "model0",
        "train",
        config=config_base_old,
        framework_fn=ReplaceVocabChecker,
    )
    new_src_vocab = "new_src_vocab.txt"
    new_tgt_vocab = "new_tgt_vocab.txt"
    tmpdir.join(new_src_vocab).write("hello\n")
    tmpdir.join(new_tgt_vocab).write("hallo\n")
    config = {
        "tokenization": {
            "source": {"vocabulary": "${TMP_DIR}/%s" % new_src_vocab},
            "target": {"vocabulary": "${TMP_DIR}/%s" % new_tgt_vocab},
        }
    }
    framework_fn = lambda: ReplaceVocabChecker(src_changed=True, tgt_changed=True)
    run_fn = lambda: _run_framework(
        tmpdir,
        "model1",
        "preprocess --build_model",
        config=config,
        parent="model0",
        env={"TMP_DIR": str(tmpdir)},
        framework_fn=framework_fn,
    )
    with pytest.raises(ValueError) as exc:
        run_fn()
    assert exc.match("replace_vocab")
    config["tokenization"]["source"]["replace_vocab"] = True
    config["tokenization"]["target"]["replace_vocab"] = True
    model_dir = run_fn()
    assert filecmp.cmp(
        os.path.join(model_dir, new_src_vocab),
        os.path.join(str(tmpdir), new_src_vocab),
        shallow=False,
    )
    assert filecmp.cmp(
        os.path.join(model_dir, new_tgt_vocab),
        os.path.join(str(tmpdir), new_tgt_vocab),
        shallow=False,
    )
    config = _read_config(model_dir)
    assert (
        config["tokenization"]["source"]["vocabulary"]
        == "${MODEL_DIR}/%s" % new_src_vocab
    )
    assert (
        config["tokenization"]["target"]["vocabulary"]
        == "${MODEL_DIR}/%s" % new_tgt_vocab
    )
    assert "replace_vocab" not in config["tokenization"]["target"]
    assert "replace_vocab" not in config["tokenization"]["source"]
    assert (
        config["tokenization"]["source"]["previous_vocabulary"]
        == "${MODEL_DIR}/previous-source-vocab.txt"
    )
    assert (
        config["tokenization"]["target"]["previous_vocabulary"]
        == "${MODEL_DIR}/previous-target-vocab.txt"
    )
    assert filecmp.cmp(
        os.path.join(model_dir, "previous-source-vocab.txt"),
        os.path.join(_test_dir(), "corpus", "vocab", "en-vocab.txt"),
        shallow=False,
    )
    assert filecmp.cmp(
        os.path.join(model_dir, "previous-target-vocab.txt"),
        os.path.join(_test_dir(), "corpus", "vocab", "de-vocab.txt"),
        shallow=False,
    )


def test_replace_vocab_new(tmpdir):
    _run_framework(
        tmpdir, "model0", "train", config=config_base, framework_fn=ReplaceVocabChecker
    )
    new_src_vocab = "new_src_vocab.txt"
    new_tgt_vocab = "new_tgt_vocab.txt"
    tmpdir.join(new_src_vocab).write("hello\n")
    tmpdir.join(new_tgt_vocab).write("hallo\n")
    config = {
        "vocabulary": {
            "source": {"path": "${TMP_DIR}/%s" % new_src_vocab},
            "target": {"path": "${TMP_DIR}/%s" % new_tgt_vocab},
        }
    }
    framework_fn = lambda: ReplaceVocabChecker(src_changed=True, tgt_changed=True)
    run_fn = lambda: _run_framework(
        tmpdir,
        "model1",
        "train",
        config=config,
        parent="model0",
        env={"TMP_DIR": str(tmpdir)},
        framework_fn=framework_fn,
    )
    with pytest.raises(ValueError) as exc:
        run_fn()
    assert exc.match("replace_vocab")
    config["vocabulary"]["source"]["replace_vocab"] = True
    config["vocabulary"]["target"]["replace_vocab"] = True
    model_dir = run_fn()
    _check_dir(
        model_dir,
        [new_src_vocab, new_tgt_vocab, "config.json", "checkpoint.txt", "checksum.md5"],
    )
    assert filecmp.cmp(
        os.path.join(model_dir, new_src_vocab),
        os.path.join(str(tmpdir), new_src_vocab),
        shallow=False,
    )
    assert filecmp.cmp(
        os.path.join(model_dir, new_tgt_vocab),
        os.path.join(str(tmpdir), new_tgt_vocab),
        shallow=False,
    )
    config = _read_config(model_dir)
    assert config["vocabulary"]["source"]["path"] == "${MODEL_DIR}/%s" % new_src_vocab
    assert config["vocabulary"]["target"]["path"] == "${MODEL_DIR}/%s" % new_tgt_vocab
    assert "replace_vocab" not in config["vocabulary"]["target"]
    assert "replace_vocab" not in config["vocabulary"]["source"]
    assert "previous_vocabulary" not in config["vocabulary"]["target"]
    assert "previous_vocabulary" not in config["vocabulary"]["source"]


def test_replace_vocab_old_to_new(tmpdir):
    _run_framework(
        tmpdir,
        "model0",
        "train",
        config=config_base_old,
        framework_fn=ReplaceVocabChecker,
    )
    new_src_vocab = "new_src_vocab.txt"
    new_tgt_vocab = "new_tgt_vocab.txt"
    tmpdir.join(new_src_vocab).write("hello\n")
    tmpdir.join(new_tgt_vocab).write("hallo\n")
    config = {
        "vocabulary": {
            "source": {"path": "${TMP_DIR}/%s" % new_src_vocab},
            "target": {"path": "${TMP_DIR}/%s" % new_tgt_vocab},
        }
    }
    framework_fn = lambda: ReplaceVocabChecker(src_changed=True, tgt_changed=True)
    run_fn = lambda: _run_framework(
        tmpdir,
        "model1",
        "train",
        config=config,
        parent="model0",
        env={"TMP_DIR": str(tmpdir)},
        framework_fn=framework_fn,
    )
    with pytest.raises(ValueError) as exc:
        run_fn()
    assert exc.match("replace_vocab")
    config["vocabulary"]["source"]["replace_vocab"] = True
    config["vocabulary"]["target"]["replace_vocab"] = True
    model_dir = run_fn()
    _check_dir(
        model_dir,
        [new_src_vocab, new_tgt_vocab, "config.json", "checkpoint.txt", "checksum.md5"],
    )
    assert filecmp.cmp(
        os.path.join(model_dir, new_src_vocab),
        os.path.join(str(tmpdir), new_src_vocab),
        shallow=False,
    )
    assert filecmp.cmp(
        os.path.join(model_dir, new_tgt_vocab),
        os.path.join(str(tmpdir), new_tgt_vocab),
        shallow=False,
    )
    config = _read_config(model_dir)
    assert config["vocabulary"]["source"]["path"] == "${MODEL_DIR}/%s" % new_src_vocab
    assert config["vocabulary"]["target"]["path"] == "${MODEL_DIR}/%s" % new_tgt_vocab
    assert "replace_vocab" not in config["vocabulary"]["target"]
    assert "replace_vocab" not in config["vocabulary"]["source"]
    assert "previous_vocabulary" not in config["vocabulary"]["target"]
    assert "previous_vocabulary" not in config["vocabulary"]["source"]


def test_replace_vocab_in_preprocess(tmpdir):
    framework_fn = lambda: ReplaceVocabChecker()
    _ = _run_framework(
        tmpdir, "model0", "train", config=config_base, framework_fn=framework_fn
    )
    new_src_vocab = "new_src_vocab.txt"
    new_tgt_vocab = "new_tgt_vocab.txt"
    tmpdir.join(new_src_vocab).write("hello\n")
    tmpdir.join(new_tgt_vocab).write("hallo\n")
    config = {
        "vocabulary": {
            "source": {"path": "${TMP_DIR}/%s" % new_src_vocab, "replace_vocab": True},
            "target": {"path": "${TMP_DIR}/%s" % new_tgt_vocab, "replace_vocab": True},
        }
    }
    model_dir = _run_framework(
        tmpdir,
        "preprocess1",
        "preprocess --build_model",
        parent="model0",
        config=config,
        env={"TMP_DIR": str(tmpdir)},
        framework_fn=framework_fn,
    )
    _check_dir(
        model_dir,
        [
            new_src_vocab,
            new_tgt_vocab,
            "config.json",
            "data",
            "checkpoint.txt",
            "checksum.md5",
            "previous-source-vocab.txt",
            "previous-target-vocab.txt",
        ],
    )
    assert filecmp.cmp(
        os.path.join(model_dir, new_src_vocab),
        os.path.join(str(tmpdir), new_src_vocab),
        shallow=False,
    )
    assert filecmp.cmp(
        os.path.join(model_dir, new_tgt_vocab),
        os.path.join(str(tmpdir), new_tgt_vocab),
        shallow=False,
    )
    assert filecmp.cmp(
        os.path.join(model_dir, "previous-source-vocab.txt"),
        os.path.join(_test_dir(), "corpus", "vocab", "en-vocab.txt"),
        shallow=False,
    )
    assert filecmp.cmp(
        os.path.join(model_dir, "previous-target-vocab.txt"),
        os.path.join(_test_dir(), "corpus", "vocab", "de-vocab.txt"),
        shallow=False,
    )
    config = _read_config(model_dir)
    assert (
        config["vocabulary"]["source"]["previous_vocabulary"]
        == "${MODEL_DIR}/previous-source-vocab.txt"
    )
    assert (
        config["vocabulary"]["target"]["previous_vocabulary"]
        == "${MODEL_DIR}/previous-target-vocab.txt"
    )
    assert config["vocabulary"]["source"]["path"] == "${MODEL_DIR}/%s" % new_src_vocab
    assert config["vocabulary"]["target"]["path"] == "${MODEL_DIR}/%s" % new_tgt_vocab
    assert "replace_vocab" not in config["vocabulary"]["target"]
    assert "replace_vocab" not in config["vocabulary"]["source"]
    framework_fn = lambda: ReplaceVocabChecker(src_changed=True, tgt_changed=True)
    model_dir = _run_framework(
        tmpdir, "model1", "train", parent="preprocess1", framework_fn=framework_fn
    )
    _check_dir(
        model_dir,
        [new_src_vocab, new_tgt_vocab, "config.json", "checkpoint.txt", "checksum.md5"],
    )
    assert filecmp.cmp(
        os.path.join(model_dir, new_src_vocab),
        os.path.join(str(tmpdir), new_src_vocab),
        shallow=False,
    )
    assert filecmp.cmp(
        os.path.join(model_dir, new_tgt_vocab),
        os.path.join(str(tmpdir), new_tgt_vocab),
        shallow=False,
    )
    config = _read_config(model_dir)
    assert config["vocabulary"]["source"]["path"] == "${MODEL_DIR}/%s" % new_src_vocab
    assert config["vocabulary"]["target"]["path"] == "${MODEL_DIR}/%s" % new_tgt_vocab
    assert "previous_vocabulary" not in config["vocabulary"]["target"]
    assert "previous_vocabulary" not in config["vocabulary"]["source"]


def test_add_new_tokens(tmpdir):
    _run_framework(
        tmpdir, "model0", "train", config=config_base, framework_fn=ReplaceVocabChecker
    )
    framework_fn = lambda: ReplaceVocabChecker(
        src_tokens_to_add=["token0"], tgt_tokens_to_add=["token1", "token2"]
    )
    model_dir = _run_framework(
        tmpdir, "model1", "train", parent="model0", framework_fn=framework_fn
    )
    _check_dir(
        model_dir,
        [
            "en-vocab.txt.v2",
            "de-vocab.txt.v2",
            "config.json",
            "checkpoint.txt",
            "checksum.md5",
        ],
    )
    config = _read_config(model_dir)
    assert config["vocabulary"]["source"]["path"] == "${MODEL_DIR}/en-vocab.txt.v2"
    assert config["vocabulary"]["target"]["path"] == "${MODEL_DIR}/de-vocab.txt.v2"
    framework_fn = lambda: ReplaceVocabChecker(src_tokens_to_add=["token3", "token4"])
    model_dir = _run_framework(
        tmpdir, "model2", "train", parent="model1", framework_fn=framework_fn
    )
    _check_dir(
        model_dir,
        [
            "en-vocab.txt.v3",
            "de-vocab.txt.v2",
            "config.json",
            "checkpoint.txt",
            "checksum.md5",
        ],
    )
    config = _read_config(model_dir)
    assert config["vocabulary"]["source"]["path"] == "${MODEL_DIR}/en-vocab.txt.v3"
    assert config["vocabulary"]["target"]["path"] == "${MODEL_DIR}/de-vocab.txt.v2"


def test_add_new_tokens_in_preprocess(tmpdir):
    _run_framework(
        tmpdir, "model0", "train", config=config_base, framework_fn=ReplaceVocabChecker
    )
    new_src_tokens = ["token0"]
    new_tgt_tokens = ["token1", "token2"]
    framework_fn = lambda: ReplaceVocabChecker(
        src_tokens_to_add=new_src_tokens, tgt_tokens_to_add=new_tgt_tokens
    )
    model_dir = _run_framework(
        tmpdir,
        "preprocess1",
        "preprocess --build_model",
        parent="model0",
        framework_fn=framework_fn,
    )
    _check_dir(
        model_dir,
        [
            "en-vocab.txt",
            "de-vocab.txt",
            "en-vocab.txt.v2",
            "de-vocab.txt.v2",
            "config.json",
            "checkpoint.txt",
            "checksum.md5",
            "data",
        ],
    )
    _check_vocab(
        os.path.join(model_dir, "en-vocab.txt.v2"),
        os.path.join(model_dir, "en-vocab.txt"),
        added_tokens=new_src_tokens,
    )
    _check_vocab(
        os.path.join(model_dir, "de-vocab.txt.v2"),
        os.path.join(model_dir, "de-vocab.txt"),
        added_tokens=new_tgt_tokens,
    )
    config = _read_config(model_dir)
    assert (
        config["vocabulary"]["source"]["previous_vocabulary"]
        == "${MODEL_DIR}/en-vocab.txt"
    )
    assert (
        config["vocabulary"]["target"]["previous_vocabulary"]
        == "${MODEL_DIR}/de-vocab.txt"
    )
    assert config["vocabulary"]["source"]["path"] == "${MODEL_DIR}/en-vocab.txt.v2"
    assert config["vocabulary"]["target"]["path"] == "${MODEL_DIR}/de-vocab.txt.v2"
    framework_fn = lambda: ReplaceVocabChecker(src_changed=True, tgt_changed=True)
    model_dir = _run_framework(
        tmpdir, "model1", "train", parent="preprocess1", framework_fn=framework_fn
    )
    _check_dir(
        model_dir,
        [
            "en-vocab.txt.v2",
            "de-vocab.txt.v2",
            "config.json",
            "checkpoint.txt",
            "checksum.md5",
        ],
    )
    config = _read_config(model_dir)
    assert "previous_vocabulary" not in config["vocabulary"]["source"]
    assert "previous_vocabulary" not in config["vocabulary"]["target"]


@pytest.mark.parametrize(
    "src_to_add,tgt_to_add,vocab_name",
    [
        (["token0"], ["token1", "token2"], "en-vocab.txt.v2"),
        (["token0"], None, "en-vocab.txt.v2"),
        (["token0"], ["token0"], "en-vocab.txt.v2"),
    ],
)
def test_add_new_tokens_joint_vocab(tmpdir, src_to_add, tgt_to_add, vocab_name):
    config = copy.deepcopy(config_base)
    source_vocab = config["vocabulary"]["source"]["path"]
    config["vocabulary"]["target"]["path"] = source_vocab
    initial_vocab_name = os.path.basename(source_vocab)
    _run_framework(
        tmpdir, "model0", "train", config=config, framework_fn=ReplaceVocabChecker
    )
    framework_fn = lambda: ReplaceVocabChecker(
        src_tokens_to_add=src_to_add, tgt_tokens_to_add=tgt_to_add, joint=True
    )
    model_dir = _run_framework(
        tmpdir, "model1", "train", parent="model0", framework_fn=framework_fn
    )
    _check_dir(model_dir, [vocab_name, "config.json", "checkpoint.txt", "checksum.md5"])
    config = _read_config(model_dir)
    assert config["vocabulary"]["source"]["path"] == "${MODEL_DIR}/%s" % vocab_name
    assert config["vocabulary"]["target"]["path"] == "${MODEL_DIR}/%s" % vocab_name

    model_dir = _run_framework(
        tmpdir,
        "preprocess1",
        "preprocess --build_model",
        parent="model0",
        framework_fn=framework_fn,
    )
    _check_dir(
        model_dir,
        [
            initial_vocab_name,
            vocab_name,
            "config.json",
            "checkpoint.txt",
            "checksum.md5",
            "data",
        ],
    )
    config = _read_config(model_dir)
    assert (
        config["vocabulary"]["source"]["previous_vocabulary"]
        == "${MODEL_DIR}/%s" % initial_vocab_name
    )
    assert (
        config["vocabulary"]["target"]["previous_vocabulary"]
        == "${MODEL_DIR}/%s" % initial_vocab_name
    )
    assert config["vocabulary"]["source"]["path"] == "${MODEL_DIR}/%s" % vocab_name
    assert config["vocabulary"]["target"]["path"] == "${MODEL_DIR}/%s" % vocab_name


def test_replace_vocab_and_add_new_tokens(tmpdir):
    _run_framework(
        tmpdir, "model0", "train", config=config_base, framework_fn=ReplaceVocabChecker
    )
    new_src_vocab = "new_src_vocab.txt"
    new_tgt_vocab = "new_tgt_vocab.txt"
    tmpdir.join(new_src_vocab).write("hello\n")
    tmpdir.join(new_tgt_vocab).write("hallo\nwie\n")
    config = {
        "vocabulary": {
            "source": {"path": "${TMP_DIR}/%s" % new_src_vocab, "replace_vocab": True},
            "target": {"path": "${TMP_DIR}/%s" % new_tgt_vocab, "replace_vocab": True},
        }
    }
    new_src_tokens = ["token0"]
    new_tgt_tokens = ["token1", "token2"]
    framework_fn = lambda: ReplaceVocabChecker(
        src_tokens_to_add=new_src_tokens,
        tgt_tokens_to_add=new_tgt_tokens,
        src_vocab_size=2,
        tgt_vocab_size=4,
    )
    model_dir = _run_framework(
        tmpdir,
        "model1",
        "train",
        parent="model0",
        config=config,
        env={"TMP_DIR": str(tmpdir)},
        framework_fn=framework_fn,
    )
    _check_dir(
        model_dir,
        [
            "%s.v2" % new_src_vocab,
            "%s.v2" % new_tgt_vocab,
            "config.json",
            "checkpoint.txt",
            "checksum.md5",
        ],
    )


def _test_translation(tmpdir, text, args=None, filename="test"):
    output_dir = tmpdir.join("output")
    output_dir.ensure(dir=1)
    output_path = str(output_dir.join("%s.de" % filename))
    input_path = str(tmpdir.join("%s.en" % filename))
    with open(input_path, "w") as input_file:
        input_file.write("%s\n" % text)
    if args is None:
        args = []
    args.extend(["--copy_source", "-i", input_path, "-o", output_path])
    _run_framework(tmpdir, "model0", "train", config=config_base)
    _run_framework(tmpdir, "model0_trans", "trans %s" % " ".join(args), parent="model0")
    copied_input_path = str(output_dir.join(os.path.basename(input_path)))
    with open(output_path) as output_file:
        target = output_file.read().strip()
    with open(copied_input_path) as copied_input_file:
        source = copied_input_file.read().strip()
    return source, target


def test_translation(tmpdir):
    source, target = _test_translation(tmpdir, "Hello world!")
    assert source == "Hello world!"
    assert target == "! world Hello"


def test_translation_no_postprocess(tmpdir):
    source, target = _test_translation(
        tmpdir, "Hello world!", args=["--no_postprocess"]
    )
    assert source == "Hello world ￭!"
    assert target == "￭! world Hello"


def test_translation_add_bt_tag(tmpdir):
    source, target = _test_translation(tmpdir, "Hello world!", args=["--add_bt_tag"])
    assert source == "Hello world!"
    assert target == "｟mrk_bt｠ ! world Hello"


def test_serve(tmpdir):
    framework = DummyFramework(stateless=True)
    _, model_info = framework.serve(config_base, None)

    request = {
        "src": [
            {"text": "Hello world!", "target_prefix": "Bonjour"},
            {"text": "How are you?", "target_prefix": "Comment"},
        ]
    }

    result = run_request(
        request,
        functools.partial(framework.forward_request, model_info),
        preprocessor=framework._get_preprocessor(config_base, train=False),
        postprocessor=framework._get_postprocessor(config_base),
        config=config_base,
    )

    # Dummy translation does "target + reversed(source)".
    assert result["tgt"][0][0]["text"] == "Bonjour! world Hello"
    assert result["tgt"][1][0]["text"] == "Comment? you are How"
