import os
import re
import random
import six

from nmtwizard import utils
from nmtwizard.logger import get_logger

logger = get_logger(__name__)

# TODO : is it ok that we sample empty lines ?

class SamplerFile(object):
    """Class to store necessary information about the sampled files."""
    def __init__(self, root, base_name, files, lines_count, no_preprocess, src_suffix, tgt_suffix):
        self.root = root
        self.base_name = base_name
        self.lines_count = lines_count
        self.files = files
        self.no_preprocess = no_preprocess
        self.src_suffix = src_suffix
        self.tgt_suffix = tgt_suffix


def count_lines(path, buffer_size=65536):
    path = utils.get_file_path(path)
    if path is None:
         logger.warning("File %s not found", path)
         return None, None
    with utils.open_file(path, "rb") as f:
        num_lines = 0
        eol = False
        while True:
            data = f.read(buffer_size)
            if not data:
                if not eol:
                    num_lines += 1
                return path, num_lines
            num_lines += data.count(b"\n")
            eol = True if data.endswith(b"\n") else False

def sample(config, source_dir):

    def _count_lines(root, base_name, annotations):
        file_path = os.path.join(root, base_name)
        files = {}
        logger.debug("Processing %s", file_path)

        # Check all directions are present and aligned, open files
        src_file, src_lines = count_lines(file_path + "." + src_suffix)
        files["src"] = src_file

        if src_file and src_lines :
            # TODO V2 : multiple sources and targets
            tgt_file, tgt_lines = count_lines(file_path + "." + tgt_suffix)
            files["tgt"] = tgt_file
            if src_lines != tgt_lines:
                logger.warning('Target file %s is not aligned with source file %s. Files will be ignored in sampling.', file_path + "." + tgt_suffix, file_path + "." + src_suffix)
                return files, 0

        files["annotations"] = {}
        for key, annot_path in annotations.items():
            for suffix in ["", src_suffix, tgt_suffix]:
                annot_file_path = os.path.join(annot_path, base_name)
                if suffix :
                    annot_file_path += "." + suffix
                annot_file, annot_lines = count_lines(annot_file_path)
                if not annot_file :
                    continue
                if suffix:
                    key = key + ":" + suffix
                files["annotations"][key] = annot_file
                if src_lines != annot_lines:
                    logger.warning('Annotation file %s is not aligned with source file %s. Files will be ignored in sampling.', annot_path, file_path + src_suffix)
                    return files, 0

        return files, src_lines

    def _discover_files():

        all_files = {}
        pattern_weights_sum = 0
        pattern_sizes = {}

        for d_idx,d_item in enumerate(sample_dist):
            # check path and ditribution
            if "path" not in d_item or "distribution" not in d_item or not isinstance(d_item['distribution'], list):
                raise ValueError('invalid distribution in collection %s' % d_item)
            if not os.path.isabs(d_item["path"]):
                if not os.path.exists(source_dir) or not os.path.isdir(source_dir):
                    raise RuntimeError('source directory %s does not exist' % source_dir)
                dpath=os.path.join(source_dir, d_item["path"])
                if not os.path.exists(dpath) or not os.path.isdir(dpath):
                    raise RuntimeError('distribution path %s does not exist' % dpath)
                d_item["path"] = dpath
            else:
                source_dir = d_item["path"]

            distribution = d_item['distribution']

            # Get annotations
            annotations = d_item.get('annotations', {})
            if not isinstance(annotations, dict):
                raise ValueError("invalid type for 'annotations' field")
            for key, annot in annotations.items():
                if not isinstance(annot, str):
                    raise ValueError("innvalid type for 'annotations' value")
                if not os.path.isabs(annot):
                    annot=os.path.join(source_dir, annot)
                    annotations[key] = annot
                if not os.path.exists(annot):
                    raise RuntimeError('annotation path %s does not exist' % annot)

            # walk over all files in path
            for root, _, files in os.walk(d_item['path']):
                for f in files:
                    src_file = os.path.join(root, f)

                    # check if filename is regular, matches main source direction and get base_name
                    if (not (os.path.isfile(src_file))) :
                        continue
                    if f.endswith(src_suffix):
                        base_name = f[:-len(src_suffix) - 1]
                    elif f.endswith(src_suffix + ".gz"):
                        base_name = f[:-len(src_suffix) - 4]
                    else:
                        continue

                    # Check all directions are present and aligned
                    # Return opened files and line count
                    opened_files, size = _count_lines(root, base_name, annotations)

                    no_preprocess = d_item.get("no_preprocess", False)

                    # build file structure
                    sampler_file = SamplerFile(root, base_name, opened_files, size, no_preprocess, src_suffix, tgt_suffix)

                    # Size is 0 if some files do not exist, cannot be aligned or empty
                    if (size == 0) :
                        continue

                    # loop over patterns in distribution, check patterns are ok and file matches one
                    for rule in distribution:
                        # distribution is a list of [pattern, weight, addtl options]
                        if (len(rule) < 2 or not isinstance(rule[0], six.string_types)) :
                            raise ValueError('invalid distribution element : %s' % rule)
                        pattern = rule[0]

                        weight = rule[1]
                        label = None
                        if isinstance(weight, six.string_types) and not weight.startswith('*'):
                            weight = float(weight)
                        if len(rule) > 2:
                            label = rule[2]
                        if pattern == '*' or re.search(pattern, base_name):
                            d_idx_pattern = str(d_idx) + "-" + pattern
                            w = {"pattern": d_idx_pattern, "weight": weight, "label": label}
                            sampler_file.weight = w
                            if d_idx_pattern not in pattern_sizes:
                                if not isinstance(weight, six.string_types):
                                    pattern_weights_sum += float(weight)
                                pattern_sizes[d_idx_pattern] = size
                            else:
                                pattern_sizes[d_idx_pattern] += size
                            break

                    # Check that the file has not been selected in another distribution
                    if base_name in all_files and \
                       hasattr(all_files[base_name], "weight") and \
                       all_files[base_name].weight is not None:
                            if hasattr(sampler_file, "weight") and sampler_file.weight is not None:
                                # Different paths in distribution produced files with the same name.
                                # This is not allowed since we write output files in the same folder.
                                raise RuntimeError('Two files with the same name %s where sampled.' % base_name)
                    else:
                        all_files[base_name] = sampler_file

        return all_files, pattern_weights_sum, pattern_sizes


    def _select_lines(f):

        sample_unique = True if 'data' not in config or 'sample_unique' not in config['data'] \
                        else config['data']['sample_unique']

        random_sample = {}

        # Unique sampling, duplicates only if oversampling.
        if not gsample or sample_unique:
            # Minimal number of occurences for each line.
                # 1  if full sample (lines_kept == lines_count or no gsample)
                # >1 if oversampling (lines_kept > lines_count)
                # 0  if undersampling (lines_kept < lines_count)
            min_occurrence = not gsample or int(f.lines_kept/f.lines_count)

            if min_occurrence:
                random_sample = {i:min_occurrence for i in range(f.lines_count)}

            # Randomly sampled additional occurences.
            if gsample:
                # Robert Floyd's algorithm for sampling without replacement.
                sampling_size = int(f.lines_kept - min_occurrence * f.lines_count)
                for d in range (f.lines_count - sampling_size, f.lines_count):
                    t = random.randint(0, d)
                    if t not in random_sample or random_sample[t] == min_occurrence:
                        random_sample[t] = random_sample.get(t, 0) + 1
                    else :
                        random_sample[d] = random_sample.get(d, 0) + 1

        # Simple random sampling, possibly with duplicates.
        else:
            for _ in range(f.lines_kept):
                i = random.randint(0, f.lines_count - 1)
                random_sample[i] = random_sample.get(i, 0) + 1

        f.random_sample = random_sample


    gsample = 0
    if 'data' in config and 'sample' in config['data'] :
        gsample = config['data']['sample']
    else :
        logger.warning('No \'sample\' size specified in configuration,'
                       'all data will be sampled.')

    # TODO V2 : multiple sources and targets
    src_suffix=config["source"]
    tgt_suffix=config["target"]

    # If there is not 'sample_dist', take uniform distribution from default data directory.
    if 'data' not in config or 'sample_dist' not in config['data'] :
        sample_dist = [ {"path": source_dir,
                         "distribution" : [["*", "1" ]] } ]
    else :
        sample_dist = config['data']['sample_dist']

    # Check and read 'sample_dist'.
    assert isinstance(sample_dist, list), "sample_dist json should be a list"

    # Find all consistent files in the directory.
    all_files, pattern_weights_sum, pattern_sizes = _discover_files()

    # In strict mode, check that all patterns have been triggered
    if 'data' in config and 'mode_strict' in config['data'] and config['data']['mode_strict']:
        for d_idx, d_item in enumerate(sample_dist):
            for rule in d_item['distribution'] :
                pattern = rule[0]
                d_idx_pattern = str(d_idx) + "-" + pattern
                if (d_idx_pattern not in pattern_sizes) :
                    raise RuntimeError('pattern \'%s\' in block %d doesn\'t match any file with strict mode enabled.' % (pattern, d_idx))


    # Adjust weights based on all sampled files for each pattern.
    weights_sum = 0
    weights_size = 0
    reserved_sample = 0
    for f in all_files.values():
        if hasattr(f, "weight") and f.weight is not None:
            lines_count = f.lines_count
            pattern = f.weight["pattern"]
            weight = f.weight["weight"]
            f.oversample = 1
            if isinstance(weight, six.string_types):
                # Oversampling with "*N"
                m = re.match(r"\*([0-9]*)$", weight)
                if not m :
                    raise RuntimeError('Wrong weight format %s for sample pattern %s.' % (weight, pattern))
                if m.groups()[0]: f.oversample = int(m.groups()[0])
                reserved_sample += lines_count * f.oversample
            else:
                file_weight = float(lines_count) / pattern_sizes[pattern]
                pattern_weight = float(f.weight["weight"]) / pattern_weights_sum
                f.weight["weight"] = file_weight * pattern_weight
                weights_sum += f.weight["weight"]
                if f.weight["weight"] != 0.0:
                    weights_size += 1
        else:
            logger.debug('No rules matching %s', f.base_name)

    # Calculate the number of lines to keep using weights and lines_counts, select lines randomly.
    distribute = max(0, gsample - reserved_sample)
    metadata = {}
    summary = {}
    leftover = 0.0
    for f in all_files.values():
        label, pattern = None, None
        f.lines_kept = 0
        if hasattr(f, "weight") and f.weight is not None:
            label = f.weight["label"]
            pattern = f.weight["pattern"]
            weight = f.weight["weight"]
            if isinstance(weight, six.string_types) or weight != 0.0:
                lines_kept = f.lines_count * f.oversample
                if gsample and not isinstance(weight, six.string_types):
                    weights_size -= 1
                    res = distribute * (weight / weights_sum)
                    leftover += res - int(res)
                    lines_kept = int(res)
                    if leftover > 1.0 :
                        lines_kept += 1
                        leftover -= 1.0
                    if weights_size == 0 and leftover > 0.5 :
                        lines_kept += 1
                f.lines_kept = lines_kept

        summary[f.base_name] = {
            "linecount" : f.lines_count,
            "linesampled" : f.lines_kept,
            "pattern" : pattern
        }
        if 'annotations' in f.files:
            summary[f.base_name]['annotations'] = list(f.files['annotations'].keys())

        metadata[f.base_name] = label

        _select_lines(f)

    return all_files.values(), summary, metadata
