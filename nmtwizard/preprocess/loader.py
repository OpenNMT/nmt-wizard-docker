# coding: utf-8
import six
import abc
import sys

from nmtwizard.preprocess import tu

@six.add_metaclass(abc.ABCMeta)
class Loader(object):
    """Base class for creating batches of TUs."""

    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError()


class BasicLoader(Loader):
    """BasicLoader class creates a one-TU batch at inference."""

    def __init__(self, basic_input, start_state):

        """ In preprocess, input is one source, untokenized and one-part.
            In postprocess, input is ((source, metadata), target), tokenized and possibly multipart."""
        self._input = basic_input
        self._start_state = start_state

    def __call__(self):
        tu_list = []
        if self._input:
            tu_list.append(tu.TranslationUnit(self._input, self._start_state))
        yield tu_list, {}
        return


class FileLoader(Loader):
    """FileLoader class creates TUs from a file or aligned files."""

    def __init__(self, input_files, start_state, batch_size = sys.maxsize):
        self._batch_size = batch_size
        self._start_state = start_state
        source_file = input_files
        target_file = None
        if isinstance(source_file, tuple):
            source_file, target_file = source_file
        if isinstance(source_file, tuple):
            source_file, self._metadata = source_file
        self._files = [open(source_file, 'r')]
        if target_file:
            self._files.append(open(target_file, 'r'))

    def close_files(self):
        for f in self._files:
            f.close()

    def __call__(self):
        tu_list = []

        # Postprocess.
        if len(self._files) > 1:
            for meta in self._metadata:

                # TODO : prefix, features
                num_parts = len(meta)
                src_lines = [next(self._files[0]).strip().split() for _ in range(num_parts)]
                tgt_lines = [next(self._files[1]).strip().split() for _ in range(num_parts)]

                tu_list.append(tu.TranslationUnit(
                    ((src_lines, meta), tgt_lines), self._start_state))

                if len(tu_list) == self._batch_size:
                    yield tu_list, {}
                    del tu_list[:] # TODO V2: Check memory usage on a big corpus

        # Preprocess.
        else :
            for line in self._files[0]:
                tu_list.append(tu.TranslationUnit(line, self._start_state))
                if len(tu_list) == self._batch_size:
                    yield tu_list, {}
                    del tu_list[:] # TODO V2: Check memory usage on a big corpus

        if tu_list:
            yield tu_list, {}
        return


class SamplerFileLoader(Loader):
    """SamplerFileLoader class creates TUs from a SamplerFile object."""

    def __init__(self, f, batch_size):
        # TODO V2: multiple src
        # Files are already opened by sampler.
        self._file = f
        self._current_line = 0
        self._batch_size = batch_size

    def close_files(self):
        # Files are closed by SamplerFile.
        self._file.close_files()

    def __call__(self):

        tu_list = []
        while True:
            del tu_list[:] # TODO V2: Check memory usage on a big corpus
            # Read sampled lines from all files and build TUs.
            batch_line = 0
            annotations = self._file.files.get("annotations", {})
            while (batch_line < self._batch_size and self._current_line < self._file.lines_count):
                src_line = self._file.files["src"].readline().strip()
                tgt_line = self._file.files["tgt"].readline().strip()
                annot_lines = {}
                for key, annot_file in annotations.items():
                    annot_lines[key] = annot_file.readline().strip()
                if (self._current_line in self._file.random_sample):
                    while self._file.random_sample[self._current_line] and \
                          batch_line < self._batch_size:
                        tu_list.append(tu.TranslationUnit((src_line, tgt_line), annotations=annot_lines))
                        batch_line += 1
                        self._file.random_sample[self._current_line] -= 1
                self._current_line += 1
            if not tu_list:
                return

            batch_meta = self._file.weight.copy()
            batch_meta["base_name"] = self._file.base_name
            batch_meta["root"] = self._file.root
            batch_meta["no_preprocess"] = self._file.no_preprocess

            yield tu_list, batch_meta
