# coding: utf-8
import six
import abc
import sys

import tu

@six.add_metaclass(abc.ABCMeta)
class Loader(object):
    """Base class for creating batches of TUs."""

    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError()


class BasicLoader(Loader):
    """BasicLoader class creates TUs from a string or a list of strings."""

    # Do we need a batch size ?
    def __call__(self, input, postprocess, src_tokenizer=None, tgt_tokenizer=None):
        tu_batch = []
        if input:
            if isinstance(input, list):
                tu_batch = [tu.TranslationUnit(i, tokenized=postprocess, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer) for i in input]
            else :
                tu_batch = [tu.TranslationUnit(input, tokenized=postprocess, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)]
        return [tu_batch]


class FileLoader(Loader):
    """FileLoader class creates TUs from a file or aligned files."""

    def __init__(self, batch_size=sys.maxsize):
        # TODO V2: multiple src
        self._batch_size = batch_size

    def open_files(self,files):
        self._file = []
        for f in files:
            self._file.append(open(f, 'r'))

    def close_files(self):
        for f in self._file:
            f.close()

    def __call__(self, postprocess, src_tokenizer=None, tgt_tokenizer=None):
        tu_batch = []
        # TODO : what we do with batch size 0 ?
        input = zip(*self._file)
        for l in input:
            src_line = l[0].strip()
            tgt_line = None
            if len(l) > 1 :
                tgt_line = l[1].strip()

            tu_batch.append(tu.TranslationUnit((src_line, tgt_line), tokenized=postprocess, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)) # TODO tokenized vs non-tokenized post/preprocessing
            if len(tu_batch) == self._batch_size:
                yield tu_batch
                del tu_batch[:] # TODO V2: Check memory usage on a big corpus

        if tu_batch:
            yield tu_batch
        return


class SamplerFileLoader(FileLoader):
    """SamplerFileLoader class creates TUs from a SamplerFile object."""

    def open_files(self,f):
        # TODO V2: multiple src
        # Files are already opened by sampler.
        self._file = f
        self._current_line = 0

    def close_files(self):
        # Files are closed by SamplerFile.
        self._file.close_files()

    def __call__(self, postprocess):

        tu_batch = []
        while True:
            del tu_batch[:] # TODO V2: Check memory usage on a big corpus
            # Read sampled lines from all files and build TUs.
            batch_line = 0
            while (batch_line < self._batch_size and self._current_line < self._file.lines_count):
                src_line = self._file.files[0].readline().strip()
                tgt_line = self._file.files[1].readline().strip()
                if (self._current_line in self._file.random_sample):
                    while self._file.random_sample[self._current_line] and \
                          batch_line < self._batch_size:
                        tu_batch.append(tu.TranslationUnit((src_line, tgt_line), tokenized=postprocess))
                        batch_line += 1
                        self._file.random_sample[self._current_line] -= 1
                self._current_line += 1
            if not tu_batch:
                return

            yield tu_batch
