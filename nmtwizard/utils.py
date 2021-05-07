"""Various utilities."""

import hashlib
import subprocess
import six
import os
import gzip

from nmtwizard.logger import get_logger

logger = get_logger(__name__)


def md5file(path):
    """Computes the MD5 hash."""
    m = hashlib.md5()
    with open(path, "rb") as f:
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
            sub_md5 = md5files(
                [
                    (os.path.join(key, filename), os.path.join(path, filename))
                    for filename in os.listdir(path)
                    if not filename.startswith(".")
                ]
            )
            m.update(six.ensure_binary(sub_md5))
        else:
            with open(path, "rb") as f:
                m.update(f.read())
    return m.hexdigest()


def run_cmd(cmd, cwd=None, background=False):
    """Runs the command."""
    logger.debug("RUN %s", " ".join(cmd))
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


def get_file_path(path):
    if os.path.isfile(path):
        return path
    elif os.path.isfile(path + ".gz"):
        return path + ".gz"
    else:
        return None


def is_gzip_file(path):
    return path.endswith(".gz")


def open_file(path, *args, **kwargs):
    if path is None:
        return None
    if is_gzip_file(path):
        return gzip.open(path, *args, **kwargs)
    else:
        return open(path, *args, **kwargs)


def count_lines(path, buffer_size=65536):
    path_new = get_file_path(path)
    if path_new is None:
        logger.warning("File %s not found", path)
        return None, None
    with open_file(path_new, "rb") as f:
        num_lines = 0
        eol = False
        while True:
            data = f.read(buffer_size)
            if not data:
                if not eol:
                    num_lines += 1
                return path_new, num_lines
            num_lines += data.count(b"\n")
            eol = True if data.endswith(b"\n") else False
