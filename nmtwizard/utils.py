"""Various utilities."""

import six
import hashlib

def md5file(fp):
    """Returns the MD5 of the file fp."""
    m = hashlib.md5()
    with open(fp, 'rb') as f:
        for l in f.readlines():
            m.update(l)
    return m.hexdigest()

def md5files(lfp):
    """Returns the MD5 of a list of key:path.

    The MD5 object is updated with: key1, file1, key2, file2, ..., keyN, fileN,
    with the keys sorted alphabetically.
    """
    m = hashlib.md5()
    sorted_lfp = sorted(lfp, key=lambda ab: ab[0])
    for ab in sorted_lfp:
        m.update(six.b(ab[0]))
        with open(ab[1], 'rb') as f:
            for l in f.readlines():
                m.update(l)
    return m.hexdigest()
