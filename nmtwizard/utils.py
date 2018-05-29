"""Various utilities."""

import hashlib
import subprocess
import six

from nmtwizard.logger import get_logger

logger = get_logger(__name__)


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
