# coding: utf-8
import six
import abc
import os

from nmtwizard import tokenizer
from nmtwizard import tu


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
                if not tu_batch :
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
        while (batch_line < self._batch_size and self._current_line < self._file.lines_count):
            src_line = self._file.files[0].readline().strip()
            tgt_line = self._file.files[1].readline().strip()
            if (self._current_line in self._file.random_sample):
                while self._file.random_sample[self._current_line] and \
                      batch_line < self._batch_size:
                    tu_batch.append(tu.TranslationUnit(src_line, tgt_line))
                    batch_line += 1
                    self._file.random_sample[self._current_line] -= 1
            self._current_line += 1


class Writer(Prepoperator):

    def __init__(self, f, preprocess_dir):

        self._preprocess_dir = preprocess_dir

        # TODO : multiple files
        # TODO : do we output ALL the files that we take as input ?
        src = os.path.join(preprocess_dir, os.path.basename(f.files[0].name))
        self._src_file_out = open(src, 'w')

        tgt = os.path.join(preprocess_dir, os.path.basename(f.files[1].name))
        self._tgt_file_out = open(tgt, 'w')

    def __call__(self, tu_batch):
        # Write lines to files from TUs
        for tu in tu_batch :
            # Write preprocessed instead of raw
            self._src_file_out.write("%s\n" % tu.src_raw)
            self._tgt_file_out.write("%s\n" % tu.tgt_raw)


class Tokenizer(Prepoperator):

    def __init__(self, tok_config):
        self._src_tokenizer = 'source' in tok_config and \
                              tokenizer.build_tokenizer(tok_config['source'])

        self._tgt_tokenizer = 'target' in tok_config and \
                              tokenizer.build_tokenizer(tok_config['target'])

    def __call__(self, tu_batch):

        for tu in tu_batch :
            if self._src_tokenizer:
                tu.src_raw = tokenizer.tokenize(self._src_tokenizer, tu.src_raw)

            if self._tgt_tokenizer:
                tu.tgt_raw = tokenizer.tokenize(self._tgt_tokenizer, tu.tgt_raw)
