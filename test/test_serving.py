import pytest

from nmtwizard import serving

def _make_output(tokens, score=None, attention=None):
    return serving.TranslationOutput(tokens, score=score, attention=attention)

def _make_example(tokens, metadata=None):
    if metadata is None:
        metadata = [None]
    return serving.TranslationExample(
        config=None,
        tokens=tokens,
        metadata=metadata)

def test_batch_iterator():
    examples = [_make_example([1]), _make_example([2]), _make_example([3])]
    assert list(serving.batch_iterator(examples)) == [[1, 2, 3]]
    assert list(serving.batch_iterator(examples, max_batch_size=4)) == [[1, 2, 3]]
    assert list(serving.batch_iterator(examples, max_batch_size=2)) == [[1, 2], [3]]

def test_preprocess_text():
    func = lambda s, _: s.split()
    example = serving.preprocess_text("a b c", func)
    assert example.tokens == [["a", "b", "c"]]
    assert example.metadata == [None]

    func = lambda s, _: (s.split(), len(s))
    example = serving.preprocess_text("a b c", func)
    assert example.tokens == [["a", "b", "c"]]
    assert example.metadata == [5]

    func = lambda s, _: ([s.split()[:2], s.split()[2:]], [1, 2])
    example = serving.preprocess_text("a b c d", func)
    assert example.tokens == [["a", "b"], ["c", "d"]]
    assert example.metadata == [1, 2]

def test_preprocess_examples():
    def func(text, config=None):
        tokens = text.split()
        if len(tokens) > 3:
            tokens = [tokens[:3], tokens[3:]]
            metadata = list(map(len, tokens))
        else:
            metadata = len(tokens)
        return tokens, metadata

    with pytest.raises(ValueError):
        serving.preprocess_examples(["a b c"], func)
    with pytest.raises(ValueError):
        serving.preprocess_examples([{"toto": "a b c"}], func)

    config = {"a": 24}
    raw_examples = [
        {"text": "a b c", "config": {"a": 42}},
        {"text": "d e f g"}
    ]

    examples = serving.preprocess_examples(raw_examples, func, config=config)
    assert len(examples) == 2
    assert examples[0].config == {"a": 42}
    assert examples[0].tokens == [["a", "b", "c"]]
    assert examples[0].metadata == [3]
    assert examples[1].config == {"a": 24}
    assert examples[1].tokens == [["d", "e", "f"], ["g"]]
    assert examples[1].metadata == [3, 1]

def test_postprocess_output():
    output = _make_output([["a", "b", "c"]], score=[2], attention=[None])
    example = _make_example([["x", "y"]], metadata=[None])

    def func(src, tgt, config=None):
        assert src == ["x", "y"]
        assert tgt == ["a", "b", "c"]
        return " ".join(src + tgt)

    result = serving.postprocess_output(output, example, func)
    assert result["text"] == "x y a b c"
    assert result["score"] == 2

def test_postprocess_output_with_metadata():
    output = _make_output([["a", "b", "c"]], score=[2], attention=[None])
    example = _make_example([["x", "y"]], metadata=[3])

    def func(src, tgt, config=None):
        assert isinstance(src, tuple)
        assert src[0] == ["x", "y"]
        assert src[1] == 3
        assert tgt == ["a", "b", "c"]
        return ""

    result = serving.postprocess_output(output, example, func)
    assert result["score"] == 2

def test_postprocess_output_multiparts():
    example = _make_example([["x", "y"], ["z"]], metadata=[3, 2])

    def func(src, tgt, config=None):
        assert isinstance(src, tuple)
        assert src[0] == [["x", "y"], ["z"]]
        assert src[1] == [3, 2]
        assert tgt == [["a", "b", "c"], ["d", "e"]]
        return ""

    output = _make_output([["a", "b", "c"], ["d", "e"]], score=[2, 6], attention=[None, None])
    result = serving.postprocess_output(output, example, func)
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

    def func(src, tgt, config=None):
        return " ".join(tgt)

    results = serving.postprocess_outputs(outputs, examples, func)
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

    def func(src, tgt, config=None):
        assert isinstance(src, tuple)
        assert src[0] == [["x", "y"], ["z"]]
        assert src[1] == [3, 2]
        assert tgt == [["a", "b", "c"], ["d", "e"]] or tgt == [["a", "c", "b"], ["e", "e"]]
        return " ".join(tgt[0] + tgt[1])

    results = serving.postprocess_outputs(outputs, examples, func)
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
    def func(batch, timeout=None):
        return [
            [_make_output(list(reversed(element)))]
            for element in batch]

    examples = [
        _make_example([["a", "b"], ["c", "d"]], metadata=[3, 2]),
        _make_example([["e", "f", "g"]], metadata=[4])
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
        serving.run_request(["abc"], None, None, None)
    with pytest.raises(ValueError):
        serving.run_request({"input": "abc"}, None, None, None)
    with pytest.raises(ValueError):
        serving.run_request({"src": "abc"}, None, None, None)

    assert serving.run_request({"src": []}, None, None, None) == {"tgt": []}

    def preprocess(src, config):
        return src.split(config["separator"])
    def translate(batch, timeout=None):
        return [
            [_make_output(list(reversed(element)))]
            for element in batch]
    def postprocess(src, tgt, config):
        return config["separator"].join(tgt)

    config = {"separator": "-"}
    request = {
        "src": [
            {"text": "a b c"},
            {"text": "x_y_z", "config": {"separator": "_"}}
        ],
        "options": {
            "config": {"separator": " "}
        }
    }

    result = serving.run_request(request, preprocess, translate, postprocess, config=config)
    assert result == {'tgt': [[{'text': 'c b a'}], [{'text': 'z_y_x'}]]}
