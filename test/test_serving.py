import pytest

from nmtwizard import serving

def _make_output(tokens, score=None, attention=None):
    return serving.TranslationOutput(tokens, score=score, attention=attention)

def _make_example(tokens, index=0, metadata=None, mode="default"):
    if metadata is None:
        metadata = [None]
    return serving.TranslationExample(
        index=index,
        config=None,
        source_tokens=tokens,
        target_tokens=[None] * len(tokens),
        mode=mode,
        metadata=metadata)

def test_batch_iterator():
    examples = [
        _make_example([1], index=0),
        _make_example([2], index=1, mode="alternatives"),
        _make_example([3, 4], index=2),
        _make_example([5], index=3, mode="alternatives"),
    ]

    sorted_batches = list(sorted(
        serving.batch_iterator(examples, max_batch_size=2),
        key=lambda batch: batch.mode))
    assert sorted_batches == [
        serving.TranslationBatch(
            indices=[1, 3],
            source_tokens=[2, 5],
            target_tokens=[None, None],
            mode="alternatives"),
        serving.TranslationBatch(
            indices=[0, 2],
            source_tokens=[1, 3],
            target_tokens=[None, None],
            mode="default"),
        serving.TranslationBatch(
            indices=[2],
            source_tokens=[4],
            target_tokens=[None],
            mode="default"),
    ]

def test_preprocess_example():

    class Processor:
        def process_input(self, source, **kwargs):
            return source.split(), None, None
    example = serving.preprocess_example(Processor(), 1, {"text": "a b c"})
    assert example.index == 1
    assert example.source_tokens == [["a", "b", "c"]]
    assert example.metadata == [None]
    assert example.mode == "default"

    class Processor:
        def process_input(self, source, target=None, **kwargs):
            return source.split(), target.split(), len(source)
    raw_example = {"text": "a b c", "target_prefix": "d e", "mode": "alternatives"}
    example = serving.preprocess_example(Processor(), 2, raw_example)
    assert example.index == 2
    assert example.source_tokens == [["a", "b", "c"]]
    assert example.target_tokens == [["d", "e"]]
    assert example.metadata == [5]
    assert example.mode == "alternatives"

    class Processor:
        def process_input(self, source, **kwargs):
            source = source.split()
            return [source[:2], source[2:]], None, [1, 2]
    example = serving.preprocess_example(Processor(), 3, {"text": "a b c d"})
    assert example.index == 3
    assert example.source_tokens == [["a", "b"], ["c", "d"]]
    assert example.metadata == [1, 2]
    assert example.mode == "default"

def test_preprocess_examples():
    class Processor:
        def process_input(self, source, **kwargs):
            tokens = source.split()
            if len(tokens) > 3:
                tokens = [tokens[:3], tokens[3:]]
                metadata = list(map(len, tokens))
            else:
                metadata = len(tokens)
            return tokens, None, metadata

    with pytest.raises(ValueError):
        serving.preprocess_examples(["a b c"], Processor())
    with pytest.raises(ValueError):
        serving.preprocess_examples([{"toto": "a b c"}], Processor())

    config = {"a": 24}
    raw_examples = [
        {"text": "a b c", "config": {"a": 42}},
        {"text": "d e f g"}
    ]

    examples = serving.preprocess_examples(raw_examples, Processor(), config=config)
    assert len(examples) == 2
    assert examples[0].config == {"a": 42}
    assert examples[0].source_tokens == [["a", "b", "c"]]
    assert examples[0].metadata == [3]
    assert examples[1].config == {"a": 24}
    assert examples[1].source_tokens == [["d", "e", "f"], ["g"]]
    assert examples[1].metadata == [3, 1]

def test_postprocess_output():
    output = _make_output([["a", "b", "c"]], score=[2], attention=[None])
    example = _make_example([["x", "y"]], metadata=[None])

    class Processor:
        def process_input(self, source, target=None, metadata=None, **kwargs):
            assert source == [["x", "y"]]
            assert target == [["a", "b", "c"]]
            assert metadata == [None]
            return " ".join(source[0] + target[0])

    result = serving.postprocess_output(output, example, Processor())
    assert result["text"] == "x y a b c"
    assert result["score"] == 2

def test_postprocess_output_with_metadata():
    output = _make_output([["a", "b", "c"]], score=[2], attention=[None])
    example = _make_example([["x", "y"]], metadata=[3])

    class Processor:
        def process_input(self, source, target=None, metadata=None, **kwargs):
            assert source == [["x", "y"]]
            assert target == [["a", "b", "c"]]
            assert metadata == [3]
            return ""

    result = serving.postprocess_output(output, example, Processor())
    assert result["score"] == 2

def test_postprocess_output_multiparts():
    example = _make_example([["x", "y"], ["z"]], metadata=[3, 2])

    class Processor:
        def process_input(self, source, target=None, metadata=None, **kwargs):
            assert source == [["x", "y"], ["z"]]
            assert target == [["a", "b", "c"], ["d", "e"]]
            assert metadata == [3, 2]
            return ""

    output = _make_output([["a", "b", "c"], ["d", "e"]], score=[2, 6], attention=[None, None])
    result = serving.postprocess_output(output, example, Processor())
    assert result["score"] == 8

def test_postprocess_outputs():
    outputs = [
        [_make_output([["a", "b", "c"]], score=[1], attention=[None]),
         _make_output([["a", "c", "b"]], score=[2], attention=[None])],
        [_make_output([["d", "e"]], score=[3], attention=[None]),
         _make_output([["e", "e"]], score=[4], attention=[None])]
    ]
    examples = [
        _make_example([["x", "y"]]),
        _make_example([["s", "t"]]),
    ]

    class Processor:
        def process_input(self, source, target=None, **kwargs):
            return " ".join(target[0])

    results = serving.postprocess_outputs(outputs, examples, Processor())
    assert len(results) == 2
    assert len(results[0]) == 2
    assert results[0][0] == {"text": "a b c", "score": 1}
    assert results[0][1] == {"text": "a c b", "score": 2}
    assert len(results[1]) == 2
    assert results[1][0] == {"text": "d e", "score": 3}
    assert results[1][1] == {"text": "e e", "score": 4}

def test_postprocess_outputs_multiparts():
    # 2 parts and 2 hypothesis.
    outputs = [[
        _make_output([["a", "b", "c"], ["d", "e"]], score=[1, 3], attention=[None, None]),
        _make_output([["a", "c", "b"], ["e", "e"]], score=[2, 4], attention=[None, None])
    ]]
    examples = [
        _make_example([["x", "y"], ["z"]], metadata=[3, 2])]

    class Processor:
        def process_input(self, source, target=None, metadata=None, **kwargs):
            assert source == [["x", "y"], ["z"]]
            assert metadata == [3, 2]
            assert (target == [["a", "b", "c"], ["d", "e"]]
                    or target == [["a", "c", "b"], ["e", "e"]])
            return " ".join(target[0] + target[1])

    results = serving.postprocess_outputs(outputs, examples, Processor())
    assert len(results) == 1
    assert len(results[0]) == 2
    assert results[0][0] == {"text": "a b c d e", "score": 1 + 3}
    assert results[0][1] == {"text": "a c b e e", "score": 2 + 4}

def test_align_tokens():
    assert serving.align_tokens([], [], []) == []
    assert serving.align_tokens(["a"], [], []) == []

    src_tokens = ["ab", "c"]
    tgt_tokens = ["1", "234", "56"]
    attention = [[0.2, 0.8], [0.7, 0.3], [0.6, 0.4]]
    alignments = serving.align_tokens(src_tokens, tgt_tokens, attention)
    assert len(alignments) == 3
    assert alignments[0] == {
        "src": [{"range": (3, 4), "id": 1}],
        "tgt": [{"range": (0, 1), "id": 0}]}
    assert alignments[1] == {
        "src": [{"range": (0, 2), "id": 0}],
        "tgt": [{"range": (2, 5), "id": 1}]}
    assert alignments[2] == {
        "src": [{"range": (0, 2), "id": 0}],
        "tgt": [{"range": (6, 8), "id": 2}]}

def test_translate_examples():
    def func(source_tokens, target_tokens, options=None):
        return [
            [_make_output(list(reversed(element)))]
            for element in source_tokens]

    examples = [
        _make_example([["a", "b"], ["c", "d"]], index=0, metadata=[3, 2]),
        _make_example([["e", "f", "g"]], index=1, metadata=[4])
    ]

    outputs = serving.translate_examples(examples, func)
    assert len(outputs) == 2
    assert len(outputs[0]) == 1
    assert outputs[0][0].output == [["b", "a"], ["d", "c"]]
    assert outputs[0][0].score == [None, None]
    assert outputs[0][0].attention == [None, None]
    assert len(outputs[1]) == 1
    assert outputs[1][0].output == [["g", "f", "e"]]
    assert outputs[1][0].score == [None]
    assert outputs[1][0].attention == [None]

def test_run_request():
    with pytest.raises(ValueError):
        serving.run_request(["abc"], None)
    with pytest.raises(ValueError):
        serving.run_request({"input": "abc"}, None)
    with pytest.raises(ValueError):
        serving.run_request({"src": "abc"}, None)

    assert serving.run_request({"src": []}, None) == {"tgt": []}

    class Preprocessor:
        def process_input(self, source, target=None, config=None, **kwargs):
            sep = config["separator"]
            source = source.split(sep)
            if target is not None:
                assert config.get("target_type") is not None
                target = target.split(sep)
            return source, target, None

    class Postprocessor:
        def process_input(self, source, target=None, config=None, **kwargs):
            return config["separator"].join(target[0])

    def translate(source_tokens, target_tokens, options=None):
        assert options is not None
        assert "config" in options  # Request options are fowarded.
        assert "mode" in options
        assert options["max_batch_size"] == 1
        return [
            [_make_output((target if target is not None else []) + list(reversed(source)))]
            for source, target in zip(source_tokens, target_tokens)]

    config = {"separator": "-"}
    request = {
        "src": [
            {"text": "a b c", "target_prefix": "1 2", "mode": "alternatives"},
            {"text": "x_y_z", "config": {"separator": "_"}}
        ],
        "options": {
            "config": {"separator": " "}
        }
    }

    result = serving.run_request(
        request,
        translate,
        Preprocessor(),
        Postprocessor(),
        config=config,
        rebatch_request=False,
        max_batch_size=1)
    assert result == {'tgt': [[{'text': '1 2 c b a'}], [{'text': 'z_y_x'}]]}
