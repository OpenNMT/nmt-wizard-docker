import six
import abc
import os

import tu


@six.add_metaclass(abc.ABCMeta)
class Prepoperator(object):
    """Base class for preprocessing opertators."""

    @abc.abstractmethod
    def __call__(self, tu_batch):
        raise NotImplementedError()

class PreprocessingPipeline(Prepoperator):

    def __init__(self):
        self._ops = []

    def add(self, op):
        self._ops.append(op)

    def __call__(self):
        tu_batch = []
        while True:
            del tu_batch[:] # TODO: Check memory usage on a big corpus
            for op in self._ops:
                op(tu_batch)
                if not len(tu_batch) :
                    return
            yield tu_batch

class Loader(Prepoperator):

    def __init__(self, f, batch_size = 0):

        # TODO : multiple src and tgt
        self._file = f

        self._batch_size = batch_size
        self._current_line = 0

    def __call__(self, tu_batch):

        # Read sampled lines from all files and build TUs.
        batch_line = 0
        while (batch_line < self._batch_size and self._current_line < self._file._linecount):
            src_line = self._file._files[0].readline()
            tgt_line = self._file._files[1].readline()
            if (self._current_line in self._file._random_sample):
                # TODO Is there a point in preprocessing duplicate lines ? Filtering ?
                # Or simply store number of occurrences for each TU.
                occurences = self._file._random_sample[self._current_line]
                tu_batch.append(tu.TranslationUnit(src_line, tgt_line, occurences))
                batch_line += 1
            self._current_line += 1


class Writer(Prepoperator):

    def __init__(self, f, preprocess_dir):

        self._preprocess_dir = preprocess_dir

        # TODO : multiple files
        # TODO : do we output ALL the files that we take as input ?
        src = os.path.join(preprocess_dir, os.path.basename(f._files[0].name))
        self._src_file_out = open(src, 'wb')

        tgt = os.path.join(preprocess_dir, os.path.basename(f._files[1].name))
        self._tgt_file_out = open(tgt, 'wb')

    def __call__(self, tu_batch):
        # Write lines to files from TUs
        for tu in tu_batch :
            for _ in range(tu._occurences) :
                # Write preprocessed instead of raw
                self._src_file_out.write(tu._src_raw)
                self._tgt_file_out.write(tu._tgt_raw)

