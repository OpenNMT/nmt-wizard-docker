import abc
import itertools

from nmtwizard import utils
from nmtwizard.preprocess import tu


class Loader(abc.ABC):
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

    def __init__(self, batch_size, batch_meta=None):
        super().__init__(batch_size)
        if batch_meta is None:
            batch_meta = {}
        self._input_paths = {}
        self._batch_meta = batch_meta

    def register_file(self, name, path):
        if name in self._input_paths:
            raise ValueError("A file is already registered with name %s" % name)
        self._input_paths[name] = path

    @abc.abstractmethod
    def _get_translation_units(self, files):
        """Yields TUs from the input files."""
        raise NotImplementedError()

    def __call__(self):
        if not self._input_paths:
            raise RuntimeError("No files have been registered")
        files = {
            name: utils.open_file(path) for name, path in self._input_paths.items()
        }

        try:
            tu_list = []

            for unit in self._get_translation_units(files):
                tu_list.append(unit)
                if len(tu_list) == self._batch_size:
                    yield tu_list, self._batch_meta.copy()
                    tu_list = []

            if tu_list:
                yield tu_list, self._batch_meta.copy()
        finally:
            for f in files.values():
                f.close()


class PreprocessFileLoader(FileLoader):
    """Loads TUs for preprocessing."""

    def __init__(self, source_path, target_path=None, batch_size=None):
        super().__init__(batch_size)
        self.register_file("source", source_path)
        if target_path is not None:
            self.register_file("target", target_path)

    def _get_translation_units(self, files):
        source_file = files["source"]
        target_file = files.get("target", itertools.repeat(None))
        for source, target in zip(source_file, target_file):
            yield tu.TranslationUnit(source=source, target=target)


class PostprocessFileLoader(FileLoader):
    """Loads TUs for postprocessing."""

    def __init__(
        self,
        source_path,
        target_path,
        metadata,
        start_state=None,
        batch_size=None,
        target_score_type=None,
    ):
        super().__init__(batch_size)
        if start_state is None:
            start_state = {}
        self._source_tokenizer = start_state.get("src_tokenizer")
        self._target_tokenizer = start_state.get("tgt_tokenizer")
        self._metadata = metadata
        self._target_score_type = target_score_type
        self.register_file("source", source_path)
        self.register_file("target", target_path)

    def _get_translation_units(self, files):
        source_file = files["source"]
        target_file = files["target"]
        for meta in self._metadata:
            # TODO : features
            num_parts = len(meta)
            src_lines = [next(source_file).strip().split(" ") for _ in range(num_parts)]
            tgt_lines = [next(target_file).strip().split(" ") for _ in range(num_parts)]

            if self._target_score_type is not None:
                score = _extract_score(tgt_lines, self._target_score_type)
                meta = [{"score": score}]

            yield tu.TranslationUnit(
                source=src_lines,
                target=tgt_lines,
                metadata=meta,
                source_tokenizer=self._source_tokenizer,
                target_tokenizer=self._target_tokenizer,
            )


def _extract_score(tokens, score_type, separator="|||"):
    total_score = 0
    total_length = 0

    for tokens_part in tokens:
        if len(tokens_part) < 2 or tokens_part[1] != separator:
            raise RuntimeError(
                "Cannot extract score from line: %s" % " ".join(tokens_part)
            )

        try:
            score = float(tokens_part[0])
        except ValueError as e:
            raise RuntimeError(
                "Cannot convert '%s' to a score in line: %s"
                % (tokens_part[0], " ".join(tokens_part))
            ) from e

        # Remove score and separator
        tokens_part.pop(0)
        tokens_part.pop(0)

        length = len(tokens_part)
        total_length += length

        # The returned score is the normalized log likelihood.
        if score_type == utils.ScoreType.NORMALIZED_NLL:
            score *= -length
        elif score_type == utils.ScoreType.NORMALIZED_LL:
            score *= length
        elif score_type == utils.ScoreType.CUMULATED_NLL:
            score = -score
        total_score += score

    return total_score / total_length if total_length != 0 else 0


class SamplerFileLoader(FileLoader):
    """SamplerFileLoader class creates TUs from a SamplerFile object."""

    def __init__(self, f, batch_size):
        # TODO V2: multiple src
        batch_meta = {
            "base_name": f.base_name,
            "label": f.label,
            "no_preprocess": f.no_preprocess,
            "pattern": f.pattern,
            "root": f.root,
            "weight": f.weight,
        }
        if f.oversample_as_weights:
            batch_meta["example_weights"] = f.oversample

        super().__init__(batch_size, batch_meta=batch_meta)
        self._file = f

        src_path = f.files["src"]
        tgt_path = f.files.get("tgt")
        annotations = f.files.get("annotations")

        self.register_file("source", src_path)
        if tgt_path is not None:
            self.register_file("target", tgt_path)
        if annotations is not None:
            for key, path in annotations.items():
                self.register_file(key, path)

    def _get_translation_units(self, files):
        src_file = files["source"]
        tgt_file = files.get("target")
        annotations = {
            key: f for key, f in files.items() if key not in ("source", "target")
        }
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
