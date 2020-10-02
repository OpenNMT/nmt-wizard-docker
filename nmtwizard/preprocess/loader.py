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

    def __init__(self, source, target=None, metadata=None, start_state=None):
        super().__init__(batch_size=1)
        self._source = source
        self._target = target
        self._metadata = metadata
        self._source_tokenizer = start_state.get('src_tokenizer') if start_state else None
        self._target_tokenizer = start_state.get('tgt_tokenizer') if start_state else None

    def __call__(self):
        tu_list = [tu.TranslationUnit(
            source=self._source,
            target=self._target,
            metadata=self._metadata,
            source_tokenizer=self._source_tokenizer,
            target_tokenizer=self._target_tokenizer)]
        yield tu_list, {}


class FileLoader(Loader):
    """FileLoader class creates TUs from a file or aligned files."""

    def __init__(self, input_files, start_state, batch_size=None):
        super().__init__(batch_size)
        self._source_tokenizer = start_state.get('src_tokenizer')
        self._target_tokenizer = start_state.get('tgt_tokenizer')
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
                        source=src_lines,
                        target=tgt_lines,
                        metadata=meta,
                        source_tokenizer=self._source_tokenizer,
                        target_tokenizer=self._target_tokenizer))

                    if len(tu_list) == self._batch_size:
                        yield tu_list, {}
                        tu_list = []

            # Preprocess.
            else :
                for line in files[0]:
                    tu_list.append(tu.TranslationUnit(source=line))
                    if len(tu_list) == self._batch_size:
                        yield tu_list, {}
                        tu_list = []

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

    def __call__(self):
        src_file = utils.open_file(self._file.files["src"])
        tgt_file = utils.open_file(self._file.files["tgt"])
        annotations = {
            key:utils.open_file(path)
            for key, path in self._file.files.get("annotations", {}).items()}

        def _get_samples():
            for i in range(self._file.lines_count):
                src_line = src_file.readline()
                tgt_line = tgt_file.readline()
                annot_lines = {}
                for key, annot_file in annotations.items():
                    annot_lines[key] = annot_file.readline()

                num_samples = self._file.random_sample.get(i, 0)
                if num_samples == 0:
                    continue

                src_line = src_line.strip()
                tgt_line = tgt_line.strip()
                for key, line in annot_lines.items():
                    annot_lines[key] = line.strip()

                while num_samples > 0:
                    yield tu.TranslationUnit(
                        source=src_line,
                        target=tgt_line,
                        annotations=annot_lines)
                    num_samples -= 1

        try:
            batch_meta = self._file.weight.copy()
            batch_meta["base_name"] = self._file.base_name
            batch_meta["root"] = self._file.root
            batch_meta["no_preprocess"] = self._file.no_preprocess

            tu_list = []

            for sample_tu in _get_samples():
                tu_list.append(sample_tu)
                if self._batch_size is not None and len(tu_list) == self._batch_size:
                    yield tu_list, batch_meta.copy()
                    tu_list = []

            if tu_list:
                yield tu_list, batch_meta.copy()
        finally:
            src_file.close()
            tgt_file.close()
            for f in annotations.values():
                f.close()


class SamplerFilesLoader(Loader):
    """Load TUs from a sequence of SamplerFile objects."""

    def __init__(self, files, batch_size):
        super().__init__(batch_size)
        self._files = files

    def __call__(self):
        for f in self._files:
            if f.lines_kept == 0:
                continue
            loader = SamplerFileLoader(f, self._batch_size)
            for tu_batch in loader():
                yield tu_batch
