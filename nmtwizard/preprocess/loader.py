# coding: utf-8
import six
import abc

from nmtwizard import utils
from nmtwizard.preprocess import tu

@six.add_metaclass(abc.ABCMeta)
class Loader(object):
    """Base class for creating batches of TUs."""

    def __init__(self, batch_size):
        self._batch_size = batch_size

    @property
    def batch_size(self):
        return self._batch_size

    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError()


class BasicLoader(Loader):
    """BasicLoader class creates a one-TU batch at inference."""

    def __init__(self, basic_input, start_state):

        """ In preprocess, input is one source, untokenized and one-part.
            In postprocess, input is ((source, metadata), target), tokenized and possibly multipart."""
        super().__init__(len(basic_input))
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

    def __init__(self, input_files, start_state, batch_size=None):
        super().__init__(batch_size)
        self._start_state = start_state
        source_file = input_files
        target_file = None
        if isinstance(source_file, tuple):
            source_file, target_file = source_file
        if isinstance(source_file, tuple):
            source_file, self._metadata = source_file
        self._files = [source_file]
        if target_file:
            self._files.append(target_file)

    def __call__(self):
        files = [utils.open_file(path) for path in self._files]

        try:
            tu_list = []

            # Postprocess.
            if len(self._files) > 1:
                if len(self._files) != 2:
                    raise RuntimeError('Should have only two files: source and target')
                if len(files[0].readlines()) != len(self._metadata):
                    raise RuntimeError('Number of lines in source file should be the same with meta data size')
                files[0].seek(0)
                for meta in self._metadata:
                    # Long sentence process only
                    # More features need to be considered
                    src_lines = [next(files[0]).strip().split()]
                    tgt_lines = [next(files[1]).strip().split()]

                    idx = 0
                    while idx < len(meta):
                        if (idx+1 < len(meta) and "continuation" in meta[idx+1] and meta[idx+1]["continuation"]):
                            local_idx = idx + 1
                            while local_idx < len(meta):
                                if( "continuation" not in meta[local_idx] or not meta[local_idx]["continuation"]):
                                    break
                                idx = idx + 1
                                local_idx = local_idx + 1
                                tgt_lines.append(next(files[1]).strip().split())
                        idx = idx + 1

                    tu_list.append(tu.TranslationUnit(
                        ((src_lines, meta), tgt_lines), self._start_state))

                    if len(tu_list) == self._batch_size:
                        yield tu_list, {}
                        del tu_list[:] # TODO V2: Check memory usage on a big corpus

            # Preprocess.
            else :
                for line in files[0]:
                    tu_list.append(tu.TranslationUnit(line, self._start_state))
                    if len(tu_list) == self._batch_size:
                        yield tu_list, {}
                        del tu_list[:] # TODO V2: Check memory usage on a big corpus

            if tu_list:
                yield tu_list, {}
        finally:
            for f in files:
                f.close()


class SamplerFileLoader(Loader):
    """SamplerFileLoader class creates TUs from a SamplerFile object."""

    def __init__(self, f, batch_size):
        # TODO V2: multiple src
        super().__init__(batch_size)
        self._file = f
        self._current_line = 0

    def __call__(self):
        src_file = utils.open_file(self._file.files["src"])
        tgt_file = utils.open_file(self._file.files["tgt"])
        annotations = {
            key:utils.open_file(path)
            for key, path in self._file.files.get("annotations", {}).items()}

        try:
            tu_list = []
            while True:
                del tu_list[:] # TODO V2: Check memory usage on a big corpus
                # Read sampled lines from all files and build TUs.
                batch_line = 0
                while ((self._batch_size is None or batch_line < self._batch_size)
                       and self._current_line < self._file.lines_count):
                    src_line = src_file.readline().strip()
                    tgt_line = tgt_file.readline().strip()
                    annot_lines = {}
                    for key, annot_file in annotations.items():
                        annot_lines[key] = annot_file.readline().strip()
                    if (self._current_line in self._file.random_sample):
                        while (self._file.random_sample[self._current_line]
                               and (self._batch_size is None or batch_line < self._batch_size)):
                            tu_list.append(tu.TranslationUnit(
                                (src_line, tgt_line),
                                annotations=annot_lines))
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
        finally:
            src_file.close()
            tgt_file.close()
            for f in annotations.values():
                f.close()
