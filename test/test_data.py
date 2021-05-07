import pytest

from nmtwizard import data


def test_paste_files(tmpdir):
    a = str(tmpdir.join("a.txt"))
    b = str(tmpdir.join("b.txt"))
    c = str(tmpdir.join("c.txt"))
    with open(a, "w") as af:
        af.write("1 2 3\n4 5\n")
    with open(b, "w") as bf:
        bf.write("7 8\n9\n")
    data.paste_files([a, b], c, separator="|")
    with open(c) as c:
        assert c.read() == "1 2 3|7 8\n4 5|9\n"
