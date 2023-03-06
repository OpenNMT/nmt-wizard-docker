"""Various utilities."""

import hashlib
import subprocess
import os
import gzip
import enum

from nmtwizard.logger import get_logger

logger = get_logger(__name__)

context_placeholder = "｟mrk_context｠"


class Task(enum.Enum):
    TRAINING = 0
    TRANSLATION = 1
    SCORING = 2


class ScoreType(enum.Enum):
    CUMULATED_LL = 0
    CUMULATED_NLL = 1
    NORMALIZED_LL = 2
    NORMALIZED_NLL = 3


def md5files(files, buffer_size=16777216):
    """Computes the combined MD5 hash of multiple files, represented as a list
    of (key, path).
    """
    m = hashlib.md5()
    for key, path in sorted(files, key=lambda x: x[0]):
        m.update(key.encode("utf-8"))
        if os.path.isdir(path):
            sub_md5 = md5files(
                [
                    (os.path.join(key, filename), os.path.join(path, filename))
                    for filename in os.listdir(path)
                    if not filename.startswith(".")
                ]
            )
            m.update(sub_md5.encode("utf-8"))
        else:
            with open(path, "rb") as f:
                while True:
                    data = f.read(buffer_size)
                    if not data:
                        break
                    m.update(data)
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


def open_and_check_unicode(path, encoding="utf-8"):
    with open_file(path, "rb") as f:
        for index, line in enumerate(f):
            try:
                yield line.decode(encoding)
            except UnicodeError as e:
                raise RuntimeError(
                    "Invalid Unicode character (shown as � below) in file '%s' on line %d:\n%s"
                    % (
                        os.path.basename(path),
                        index + 1,
                        line.decode(encoding, errors="replace").strip(),
                    )
                ) from e


def count_lines(path, buffer_size=65536):
    with open_file(path, "rb") as f:
        num_lines = 0
        eol = False
        while True:
            data = f.read(buffer_size)
            if not data:
                if not eol:
                    num_lines += 1
                return num_lines
            num_lines += data.count(b"\n")
            eol = True if data.endswith(b"\n") else False
