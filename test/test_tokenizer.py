import pytest

from nmtwizard.preprocess import tokenizer


def test_vocabulary_iterator(tmpdir):
    vocab_path = str(tmpdir.join("vocab.txt"))
    with open(vocab_path, "w") as vocab_file:
        vocab_file.write("# Comment 1\n")
        vocab_file.write("# Comment 2\n")
        vocab_file.write("\n")
        vocab_file.write("hello\n")
        vocab_file.write("world 42\n")
        vocab_file.write("toto 0.0224656\n")
        vocab_file.write("titi 2.8989e-08\n")
        vocab_file.write("hello world\n")  # Bad token with a space.

    tokens = list(tokenizer.vocabulary_iterator(vocab_path))
    assert tokens == ["", "hello", "world", "toto", "titi", "hello world"]
