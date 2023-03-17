import abc
import json
import itertools
import os
import re
import random

import json_stream

from nmtwizard import utils
from nmtwizard.logger import get_logger

logger = get_logger(__name__)

# TODO : is it ok that we sample empty lines ?


class SamplerFile(object):
    """Class to store necessary information about the sampled files."""

    def __init__(
        self,
        root,
        base_name,
        reader,
        lines_count,
        no_preprocess,
        src_suffix,
        tgt_suffix,
        pattern,
        weight,
        label,
    ):
        self.root = root
        self.base_name = base_name
        self.lines_count = lines_count
        self.lines_kept = 0
        self.oversample = 1
        self.oversample_as_weights = False
        self.reader = reader
        self.no_preprocess = no_preprocess
        self.src_suffix = src_suffix
        self.tgt_suffix = tgt_suffix
        self.pattern = pattern
        self.weight = weight
        self.label = label


def sample(config, source_dir, oversample_as_weights):
    def _discover_files(source_dir):

        all_files = {}
        pattern_weights_sum = 0
        pattern_sizes = {}

        for d_idx, d_item in enumerate(sample_dist):
            # check path and ditribution
            if (
                "path" not in d_item
                or "distribution" not in d_item
                or not isinstance(d_item["distribution"], list)
            ):
                raise ValueError("invalid distribution in collection %s" % d_item)
            if not os.path.isabs(d_item["path"]):
                if not os.path.exists(source_dir) or not os.path.isdir(source_dir):
                    raise RuntimeError(
                        "source directory %s does not exist" % source_dir
                    )
                dpath = os.path.join(source_dir, d_item["path"])
                if not os.path.exists(dpath) or not os.path.isdir(dpath):
                    raise RuntimeError("distribution path %s does not exist" % dpath)
                d_item["path"] = dpath
            else:
                source_dir = d_item["path"]

            distribution = d_item["distribution"]

            # Get annotations
            annotations = d_item.get("annotations", {})
            if not isinstance(annotations, dict):
                raise ValueError("invalid type for 'annotations' field")
            for key, annot in annotations.items():
                if not isinstance(annot, str):
                    raise ValueError("innvalid type for 'annotations' value")
                if not os.path.isabs(annot):
                    annot = os.path.join(source_dir, annot)
                    annotations[key] = annot
                if not os.path.exists(annot):
                    raise RuntimeError("annotation path %s does not exist" % annot)

            # walk over all files in path
            for root, _, files in os.walk(d_item["path"]):
                for f in files:
                    src_file = os.path.join(root, f)

                    # check if filename is regular, matches main source direction and get base_name
                    if not (os.path.isfile(src_file)):
                        continue

                    base_name = f
                    if utils.is_gzip_file(base_name):
                        base_name = os.path.splitext(base_name)[0]

                    if base_name.endswith(".json"):
                        base_name = os.path.splitext(base_name)[0]
                        reader = JsonReader(root, base_name)
                    elif base_name.endswith(src_suffix):
                        base_name = os.path.splitext(base_name)[0]
                        base_path = os.path.join(root, base_name)
                        reader = BitextReader(base_path, src_suffix, tgt_suffix)
                    else:
                        continue

                    if annotations:
                        reader = ReaderWithExternalAnnotations(
                            reader, annotations, src_suffix, tgt_suffix
                        )

                    sampler_file = None
                    # loop over patterns in distribution, check patterns are ok and file matches one
                    for rule in distribution:
                        # distribution is a list of [pattern, weight, addtl options]
                        if (
                            len(rule) < 2
                            or len(rule) > 3
                            or not isinstance(rule[0], str)
                        ):
                            raise ValueError("invalid distribution element : %s" % rule)
                        pattern = rule[0]

                        weight = rule[1]
                        label = None
                        overweight_factor = 1
                        if isinstance(weight, list):
                            overweight_factor = weight[1]
                            weight = weight[0]
                            if isinstance(weight, str) and weight.startswith("*"):
                                raise ValueError(
                                    "Cannot add overweight factor to '%s' weight for pattern '%s'."
                                    % (weight, pattern)
                                )
                        if isinstance(weight, str) and not weight.startswith("*"):
                            weight = float(weight)
                        if len(rule) > 2:
                            label = rule[2]
                        if pattern == "*" or re.search(pattern, base_name):
                            d_idx_pattern = str(d_idx) + "-" + pattern
                            # Check all directions are present and aligned
                            size = reader.count_lines()

                            # Size is 0 if some files do not exist, cannot be aligned or empty
                            if size == 0:
                                break

                            # build file structure
                            sampler_file = SamplerFile(
                                root,
                                base_name,
                                reader,
                                size,
                                d_item.get("no_preprocess", False),
                                src_suffix,
                                tgt_suffix,
                                d_idx_pattern,
                                weight,
                                label,
                            )

                            sampler_file.oversample *= overweight_factor

                            if d_idx_pattern not in pattern_sizes:
                                if not isinstance(weight, str):
                                    pattern_weights_sum += float(weight)
                                pattern_sizes[d_idx_pattern] = size
                            else:
                                pattern_sizes[d_idx_pattern] += size
                            break

                    if sampler_file is None:
                        continue
                    # Check that the file has not been selected in another distribution
                    if base_name in all_files:
                        first_file = all_files[base_name]
                        # Different paths in distribution produced files with the same name.
                        # This is not allowed since we write output files in the same folder.
                        raise RuntimeError(
                            "Two files with the same name were sampled: '%s'.\n"
                            "First file found in path '%s', triggered by pattern '%s'.\n"
                            "Second file found in path '%s', triggered by pattern '%s'."
                            % (
                                base_name,
                                first_file.root,
                                first_file.pattern,
                                sampler_file.root,
                                sampler_file.pattern,
                            )
                        )
                    all_files[base_name] = sampler_file

        return all_files, pattern_weights_sum, pattern_sizes

    def _select_lines(f):

        sample_unique = (
            True
            if "data" not in config or "sample_unique" not in config["data"]
            else config["data"]["sample_unique"]
        )

        random_sample = {}

        # Unique sampling, duplicates only if oversampling.
        if not gsample or sample_unique:
            # Minimal number of occurences for each line.
            # 1  if full sample (lines_kept == lines_count or no gsample)
            # >1 if oversampling (lines_kept > lines_count)
            # 0  if undersampling (lines_kept < lines_count)
            min_occurrence = int(f.lines_kept / f.lines_count) or int(not gsample)

            if min_occurrence:
                random_sample = {i: min_occurrence for i in range(f.lines_count)}

            # Randomly sampled additional occurences.
            if gsample:
                # Robert Floyd's algorithm for sampling without replacement.
                sampling_size = int(f.lines_kept - min_occurrence * f.lines_count)
                for d in range(f.lines_count - sampling_size, f.lines_count):
                    t = random.randint(0, d)
                    if t not in random_sample or random_sample[t] == min_occurrence:
                        random_sample[t] = random_sample.get(t, 0) + 1
                    else:
                        random_sample[d] = random_sample.get(d, 0) + 1

        # Simple random sampling, possibly with duplicates.
        else:
            for _ in range(f.lines_kept):
                i = random.randint(0, f.lines_count - 1)
                random_sample[i] = random_sample.get(i, 0) + 1

        f.random_sample = random_sample

    gsample = config.get("data", {}).get("sample")
    if gsample is None:
        logger.warning(
            "No 'sample' size specified in configuration," "all data will be sampled."
        )
        gsample = 0

    if gsample == 0:
        logger.info("Sampling all data...")
    else:
        logger.info("Sampling %d lines...", gsample)

    # TODO V2 : multiple sources and targets
    src_suffix = config["source"]
    tgt_suffix = config.get("target", None)

    # If there is not 'sample_dist', take uniform distribution from default data directory.
    if "data" not in config or "sample_dist" not in config["data"]:
        sample_dist = [{"path": source_dir, "distribution": [["*", "1"]]}]
    else:
        sample_dist = config["data"]["sample_dist"]

    # Check and read 'sample_dist'.
    assert isinstance(sample_dist, list), "sample_dist json should be a list"

    # Find all consistent files in the directory.
    all_files, pattern_weights_sum, pattern_sizes = _discover_files(source_dir)

    # In strict mode, check that all patterns have been triggered
    for d_idx, d_item in enumerate(sample_dist):
        if d_item.get("mode_strict"):
            patterns = set()
            for rule in d_item["distribution"]:
                pattern = rule[0]
                if pattern in patterns:
                    raise RuntimeError(
                        "pattern '%s' in block %d appears more than once, with strict mode enabled."
                        % (pattern, d_idx)
                    )
                else:
                    patterns.add(pattern)
                d_idx_pattern = str(d_idx) + "-" + pattern
                if d_idx_pattern not in pattern_sizes:
                    raise RuntimeError(
                        "pattern '%s' in block %d doesn't match any file with strict mode enabled."
                        % (pattern, d_idx)
                    )

    # Adjust weights based on all sampled files for each pattern.
    weights_sum = 0
    weights_size = 0
    reserved_sample = 0
    add_example_weights = False
    for f in all_files.values():
        if isinstance(f.weight, str):
            # Oversampling with "*N"
            m = re.match(r"\*([0-9]*)([ws]?)$", f.weight)
            if not m:
                raise RuntimeError(
                    "Wrong weight format %s for sample pattern %s."
                    % (f.weight, f.pattern)
                )
            match_groups = m.groups()
            if match_groups[0]:
                f.oversample = int(match_groups[0])
            added_sample = f.lines_count
            f.oversample_as_weights = oversample_as_weights
            if match_groups[1]:
                if match_groups[1] == "w":
                    f.oversample_as_weights = True
                elif match_groups[1] == "s":
                    f.oversample_as_weights = False
            if not f.oversample_as_weights:
                added_sample *= f.oversample
            else:
                add_example_weights = True
            reserved_sample += added_sample
        else:
            if f.oversample > 1:
                f.oversample_as_weights = True
                add_example_weights = True
            file_weight = f.lines_count / pattern_sizes[f.pattern]
            pattern_weight = (
                f.weight / pattern_weights_sum if pattern_weights_sum else 0
            )
            f.weight = file_weight * pattern_weight
            weights_sum += f.weight
            if f.weight != 0.0:
                weights_size += 1

    # Calculate the number of lines to keep using weights and lines_counts, select lines randomly.
    distribute = max(0, gsample - reserved_sample)
    summary = {}
    leftover = 0.0
    logger.info("Summary of sampled lines:")
    for f in all_files.values():
        if isinstance(f.weight, str) or f.weight != 0.0:
            lines_kept = f.lines_count
            if not f.oversample_as_weights:
                lines_kept *= f.oversample
                f.oversample = 1
            if add_example_weights:
                f.oversample_as_weights = True
            if gsample and not isinstance(f.weight, str):
                weights_size -= 1
                res = distribute * (f.weight / weights_sum)
                leftover += res - int(res)
                lines_kept = int(res)
                if leftover > 1.0:
                    lines_kept += 1
                    leftover -= 1.0
                if weights_size == 0 and leftover > 0.5:
                    lines_kept += 1
            f.lines_kept = lines_kept

            logger.info(
                "\t%s: %d (out of %d) from '%s'",
                f.base_name,
                f.lines_kept,
                f.lines_count,
                f.pattern,
            )
            summary[f.base_name] = {
                "linecount": f.lines_count,
                "linesampled": f.lines_kept,
                "pattern": f.pattern,
                **f.reader.get_summary(),
            }

        _select_lines(f)

    return all_files.values(), summary


class CorpusReader(abc.ABC):
    """Base class for reading a corpus."""

    def __init__(self, base_name):
        self.base_name = base_name

    @abc.abstractmethod
    def count_lines(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def read_lines(self):
        raise NotImplementedError()

    def get_summary(self):
        return {}


class JsonReader(CorpusReader):
    """Read segments from a JSON file with a custom structure."""

    def __init__(self, root, base_name):
        super().__init__(base_name)
        self.json_path = os.path.join(root, "%s.json" % base_name)
        self.metadata_path = os.path.join(root, ".%s.metadata" % base_name)

    def count_lines(self):
        with open(self.metadata_path) as metadata_file:
            metadata = json.load(metadata_file)
            return sum(int(f["nbSegments"]) for f in metadata["files"])

    def read_lines(self):
        with utils.open_file(self.json_path, "rt") as json_file:
            data = json_stream.load(json_file)

            for segment in data["segments"].persistent():
                src_line = segment["seg"]
                tgt_line = segment["tgts"][0]["seg"]
                annotations = segment["tgts"][0].get("metadata", {})
                yield src_line, tgt_line, {**annotations}


class BitextReader(CorpusReader):
    """Read parallel source and target text files."""

    def __init__(self, base_path, src_lang, tgt_lang):
        super().__init__(os.path.basename(base_path))
        self.base_path = base_path

        self.src_path = "%s.%s" % (base_path, src_lang)
        self.src_real_path = utils.get_file_path(self.src_path)

        if tgt_lang:
            self.tgt_path = "%s.%s" % (base_path, tgt_lang)
            self.tgt_real_path = utils.get_file_path(self.tgt_path)
        else:
            self.tgt_path = None
            self.tgt_real_path = None

    def count_lines(self):
        logger.debug("Counting lines in corpus %s", self.base_path)

        if self.src_real_path is None:
            logger.warning(
                "The source file %s does not exist and will be ignored in sampling.",
                self.src_path,
            )
            return 0

        # Check all directions are present and aligned
        src_lines = utils.count_lines(self.src_real_path)

        if self.tgt_path is not None:
            if self.tgt_real_path is None:
                logger.warning(
                    "Target file %s does not exist. The source file %s will be ignored in sampling.",
                    self.tgt_path,
                    self.src_path,
                )
                return 0

            tgt_lines = utils.count_lines(self.tgt_real_path)

            if src_lines != tgt_lines:
                logger.warning(
                    "Target file %s (%d lines) is not aligned with source file "
                    "%s (%d lines). Files will be ignored in sampling.",
                    self.tgt_path,
                    tgt_lines,
                    self.src_path,
                    src_lines,
                )
                return 0

        return src_lines

    def read_lines(self):
        src_lines = utils.open_and_check_unicode(self.src_real_path)
        tgt_lines = (
            utils.open_and_check_unicode(self.tgt_real_path)
            if self.tgt_path is not None
            else itertools.repeat(None)
        )

        for src_line, tgt_line in zip(src_lines, tgt_lines):
            annotations = {}
            yield src_line, tgt_line, annotations


class ReaderWithExternalAnnotations(CorpusReader):
    """Read annotations from parallel text files."""

    def __init__(self, reader, annotations, src_lang, tgt_lang):
        super().__init__(reader.base_name)
        self.reader = reader
        self.annotations_path = {}

        for suffix in ("", src_lang, tgt_lang):
            for key, annot_path in annotations.items():
                annot_path = os.path.join(annot_path, self.base_name)

                if suffix:
                    annot_path += "." + suffix
                    key = key + ":" + suffix

                annot_path = utils.get_file_path(annot_path)
                if annot_path is None:
                    continue

                self.annotations_path[key] = annot_path

    def count_lines(self):
        num_lines = self.reader.count_lines()

        for annot_path in self.annotations_path.values():
            annot_lines = utils.count_lines(annot_path)

            if annot_lines != num_lines:
                logger.warning(
                    "Annotation file %s (%d lines) is not aligned with corpus "
                    "%s (%d lines). Files will be ignored in sampling.",
                    annot_path,
                    annot_lines,
                    self.base_name,
                    num_lines,
                )
                return 0

        return num_lines

    def read_lines(self):
        annotations_lines = {
            key: utils.open_and_check_unicode(path)
            for key, path in self.annotations_path.items()
        }

        for src_line, tgt_line, annotations in self.reader.read_lines():
            annotations.update(
                {key: next(lines) for key, lines in annotations_lines.items()}
            )

            yield src_line, tgt_line, annotations

    def get_summary(self):
        summary = self.reader.get_summary()
        summary["annotations"] = list(self.annotations_path.keys())
        return summary
