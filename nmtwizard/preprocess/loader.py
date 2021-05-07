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


class FileLoader(Loader):
    """FileLoader class creates TUs from a file or aligned files."""

    def __init__(
        self,
        source_file,
        target_file=None,
        metadata=None,
        start_state=None,
        batch_size=None,
    ):
        super().__init__(batch_size)
        if start_state is None:
            start_state = {}
        self._source_tokenizer = start_state.get("src_tokenizer")
        self._target_tokenizer = start_state.get("tgt_tokenizer")
        self._metadata = metadata
        self._files = [source_file]
        if target_file:
            self._files.append(target_file)

    def __call__(self):
        files = [utils.open_file(path) for path in self._files]

        try:
            tu_list = []

            # Postprocess.
            if len(self._files) > 1:
                for meta in self._metadata:

                    # TODO : prefix, features
                    num_parts = len(meta)
                    src_lines = [
                        next(files[0]).strip().split() for _ in range(num_parts)
                    ]
                    tgt_lines = [
                        next(files[1]).strip().split() for _ in range(num_parts)
                    ]

                    tu_list.append(
                        tu.TranslationUnit(
                            source=src_lines,
                            target=tgt_lines,
                            metadata=meta,
                            source_tokenizer=self._source_tokenizer,
                            target_tokenizer=self._target_tokenizer,
                        )
                    )

                    if len(tu_list) == self._batch_size:
                        yield tu_list, {}
                        tu_list = []

            # Preprocess.
            else:
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

    def __init__(self, f, batch_size, oversample_as_weights):
        # TODO V2: multiple src
        super().__init__(batch_size)
        self._file = f
        self._oversample_as_weights = oversample_as_weights

    def __call__(self):
        src_file = utils.open_file(self._file.files["src"])
        tgt_file = utils.open_file(self._file.files.get("tgt", None))
        annotations = {
            key: utils.open_file(path)
            for key, path in self._file.files.get("annotations", {}).items()
        }

        def _get_samples():
            for i in range(self._file.lines_count):
                src_line = src_file.readline()
                tgt_line = tgt_file.readline() if tgt_file else None
                annot_lines = {}
                for key, annot_file in annotations.items():
                    annot_lines[key] = annot_file.readline()

                num_samples = self._file.random_sample.get(i, 0)
                if num_samples == 0:
                    continue

                src_line = src_line.strip()
                if tgt_line:
                    tgt_line = tgt_line.strip()
                for key, line in annot_lines.items():
                    annot_lines[key] = line.strip()

                while num_samples > 0:
                    yield tu.TranslationUnit(
                        source=src_line, target=tgt_line, annotations=annot_lines
                    )
                    num_samples -= 1

        try:
            batch_meta = {
                "base_name": self._file.base_name,
                "label": self._file.label,
                "no_preprocess": self._file.no_preprocess,
                "pattern": self._file.pattern,
                "root": self._file.root,
                "weight": self._file.weight,
            }

            if self._oversample_as_weights:
                batch_meta["example_weights"] = self._file.oversample

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
            if tgt_file:
                tgt_file.close()
            for f in annotations.values():
                f.close()


class SamplerFilesLoader(Loader):
    """Load TUs from a sequence of SamplerFile objects."""

    def __init__(self, files, batch_size, oversample_as_weights):
        super().__init__(batch_size)
        self._files = files
        self._oversample_as_weights = oversample_as_weights

    def __call__(self):
        for f in self._files:
            if f.lines_kept == 0:
                continue
            loader = SamplerFileLoader(f, self._batch_size, self._oversample_as_weights)
            for tu_batch in loader():
                yield tu_batch
