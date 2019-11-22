"""Various utilities."""

import hashlib
import subprocess
import six
import os

from nmtwizard.logger import get_logger

logger = get_logger(__name__)


def md5file(path):
    """Computes the MD5 hash."""
    m = hashlib.md5()
    with open(path, 'rb') as f:
        m.update(f.read())
    return m.hexdigest()

def md5files(files):
    """Computes the combined MD5 hash of multiple files, represented as a list
    of (key, path).
    """
    m = hashlib.md5()
    for key, path in sorted(files, key=lambda x: x[0]):
        m.update(six.ensure_binary(key))
        if os.path.isdir(path):
            sub_md5 = md5files([
                (os.path.join(key, filename), os.path.join(path, filename))
                for filename in os.listdir(path)
                if not filename.startswith('.')])
            m.update(six.ensure_binary(sub_md5))
        else:
            with open(path, 'rb') as f:
                m.update(f.read())
    return m.hexdigest()

def run_cmd(cmd, cwd=None, background=False):
    """Runs the command."""
    logger.debug('RUN %s', ' '.join(cmd))
    if background:
        return subprocess.Popen(cmd, cwd=cwd)
    else:
        return subprocess.call(cmd, cwd=cwd)

def count_devices(gpuid):
    if isinstance(gpuid, list):
        return len(gpuid)
    else:
        return 1

def pad_lists(lists, padding_value=None, max_length=None):
    """Pads a list of lists.

    Args:
      lists: A list of lists.

    Returns:
      A tuple with the padded collection of lists and the original length of each
      list.
    """
    if max_length is None:
        max_length = max(len(lst) for lst in lists)
    lengths = []
    for lst in lists:
        length = len(lst)
        lst += [padding_value] * (max_length - length)
        lengths.append(length)
    return lists, lengths
