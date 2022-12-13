import abc
import itertools
import random

from nmtwizard import utils
from nmtwizard.logger import get_logger
from nmtwizard.preprocess import tu

logger = get_logger(__name__)


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


def _add_line_to_context(context, line, length):
    if line is not None and line.strip():
        context.append(line)
        if len(context) > length:
            del context[0]
    else:
        context.clear()


class FileLoader(Loader):
    """FileLoader class creates TUs from a file or aligned files."""

    def __init__(self, batch_size, batch_meta=None, context=None):
        super().__init__(batch_size)
        if batch_meta is None:
            batch_meta = {}
        self._input_paths = {}
        self._batch_meta = batch_meta
        if not isinstance(context, dict):
            context = {}
        self._context_prob = context.get("prob")
        self._context_length = context.get("length")
        self._context_target = context.get("target")
        self._context_labels = context.get("labels")
        self._context_apply_in_inference = context.get("apply_in_inference", False)

    def _add_context_to_tu(self, context, tu, side):
        for i, c in enumerate(reversed(context)):
            if isinstance(c, bytes):
                c = c.decode("utf-8")  # Ensure Unicode string.
            if side == "source":
                tu.add_source(
                    c,
                    name="context_" + str(i),
                    output_delimiter=utils.context_placeholder,
                    before_main=True,
                )
            elif side == "target":
                tu.add_target(
                    c,
                    name="context_" + str(i),
                    output_delimiter=utils.context_placeholder,
                    before_main=True,
                )

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

    def __init__(self, source_path, target_path=None, batch_size=None, context=None):
        super().__init__(batch_size, context=context)
        self.register_file("source", source_path)
        if target_path is not None:
            self.register_file("target", target_path)

    def _get_translation_units(self, files):
        source_file = files["source"]
        target_file = files.get("target", itertools.repeat(None))
        source_context = []
        target_context = []
        for source, target in zip(source_file, target_file):
            current_tu = tu.TranslationUnit(source=source, target=target)
            if self._context_length is not None and self._context_apply_in_inference:
                if source_context and source.strip():
                    self._add_context_to_tu(source_context, current_tu, side="source")
                _add_line_to_context(source_context, source, self._context_length)
                if self._context_target:
                    if target_context and target.strip():
                        self._add_context_to_tu(
                            target_context, current_tu, side="target"
                        )
                    _add_line_to_context(target_context, target, self._context_length)
            yield current_tu


def _make_tokens_iterator(input_file):
    for line in input_file:
        yield line.strip().split(" ")


class PostprocessFileLoader(FileLoader):
    """Loads TUs for postprocessing."""

    def __init__(
        self,
        source_path,
        target_path,
        metadata=None,
        start_state=None,
        batch_size=None,
        target_score_type=None,
    ):
        super().__init__(batch_size)
        if start_state is None:
            start_state = {}
        self._source_tokenizer = start_state.get("src_tokenizer")
        self._target_tokenizer = start_state.get("tgt_tokenizer")
        self._target_score_type = target_score_type
        self.register_file("source", source_path)
        self.register_file("target", target_path)

        _, source_size = utils.count_lines(source_path)
        _, target_size = utils.count_lines(target_path)

        self._num_hypotheses = (
            target_size // source_size
            if target_size > source_size and target_size % source_size == 0
            else 1
        )

        if not metadata:
            metadata = itertools.repeat([None], source_size)
        self._metadata = metadata

    def _get_parts(self, source_file, target_file):
        source_file = _make_tokens_iterator(source_file)
        target_file = _make_tokens_iterator(target_file)
        for metadata in self._metadata:
            num_parts = len(metadata)
            num_hyps = self._num_hypotheses
            src_lines = [next(source_file) for _ in range(num_parts)]
            tgt_lines = [next(target_file) for _ in range(num_parts * num_hyps)]

            for i in range(num_hyps):
                yield metadata.copy(), src_lines.copy(), tgt_lines[i::num_hyps]

    def _get_translation_units(self, files):
        source_file = files["source"]
        target_file = files["target"]
        for meta, src_lines, tgt_lines in self._get_parts(source_file, target_file):
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

    def __init__(self, f, batch_size, context=None):
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

        super().__init__(batch_size, batch_meta=batch_meta, context=context)
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
        def add_context_ph_to_vocab(side):
            self._batch_meta.setdefault("tokens_to_add", {})
            self._batch_meta["tokens_to_add"].setdefault(side, set())
            self._batch_meta["tokens_to_add"][side].add(utils.context_placeholder)

        src_file = files["source"]
        tgt_file = files.get("target")
        annotations = {
            key: f for key, f in files.items() if key not in ("source", "target")
        }
        source_context = []
        target_context = []
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

            tu_source_context = None
            tu_target_context = None
            if self._context_prob is not None and random.random() <= self._context_prob:
                tu_source_context = source_context
                tu_target_context = target_context
            while num_samples > 0:
                current_tu = tu.TranslationUnit(
                    source=src_line,
                    target=tgt_line,
                    annotations=annot_lines,
                )
                if tu_source_context and src_line.strip():
                    self._add_context_to_tu(source_context, current_tu, side="source")
                    if not hasattr(self, "_add_to_source_vocab"):
                        self._add_to_source_vocab = True
                if tu_target_context and tgt_line.strip():
                    self._add_context_to_tu(target_context, current_tu, side="target")
                    if not hasattr(self, "_add_to_target_vocab"):
                        self._add_to_target_vocab = True
                yield current_tu
                num_samples -= 1

            if self._context_length is not None:
                batch_labels = self._batch_meta["label"]
                if (
                    self._context_labels is None
                    or (
                        isinstance(batch_labels, list)
                        and any(label in self._context_labels for label in batch_labels)
                    )
                    or batch_labels in self._context_labels
                ):
                    _add_line_to_context(source_context, src_line, self._context_length)
                    if self._context_target:
                        _add_line_to_context(
                            target_context, tgt_line, self._context_length
                        )

        if hasattr(self, "_add_to_source_vocab") and self._add_to_source_vocab:
            add_context_ph_to_vocab("source")
            self._add_to_source_vocab = False
        if hasattr(self, "_add_to_target_vocab") and self._add_to_target_vocab:
            add_context_ph_to_vocab("target")
            self._add_to_target_vocab = False


class SamplerFilesLoader(Loader):
    """Load TUs from a sequence of SamplerFile objects."""

    def __init__(self, files, batch_size, context=None):
        super().__init__(batch_size)
        self._files = files
        self._context = context

    def __call__(self):
        for f in self._files:
            if f.lines_kept == 0:
                continue
            loader = SamplerFileLoader(f, self._batch_size, context=self._context)
            for tu_batch in loader():
                yield tu_batch
