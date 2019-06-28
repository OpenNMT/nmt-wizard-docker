import os
import re
import shutil
import random
import collections
import gzip
import six
import errno

from nmtwizard.logger import get_logger

logger = get_logger(__name__)

class SamplerFile:
    """Class to store necessary information about the sampled files."""
    def __init__(self, fullpathdir, basename, files, linecount = 0):
        self._fullpathdir = fullpathdir
        self._basename = basename
        self._linecount = linecount
        self._files = files

    def close_files(self) :
        for f in self._files :
            if not f.closed:
                f.close()

def count_lines(path):
    f = None
    i = 0
    if os.path.isfile(path + ".gz"):
        f = gzip.open(path + ".gz", 'r')
    elif os.path.isfile(path):
        f = open(path, 'r')
    if f is not None:
        for i, _ in enumerate(f):
            pass
        f.seek(0)
    return f, i + 1

def sample(config, source_dir):

    def _countLines(filepath):
        files = []
        logger.debug("Processing %s", filepath)

        # Check all directions are present and aligned, open files
        src_file, src_lines = count_lines(filepath + src_suffix)
        files.append(src_file)

        if src_file and src_lines :
            # TODO : multiple sources and targets
            tgt_file, tgt_lines = count_lines(filepath + tgt_suffix)
            files.append(tgt_file)
            if src_lines != tgt_lines:
                return files, 0

        return files, src_lines

    def _discover_files():

        allfiles = []
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

            distribution = d_item['distribution']
            # walk over all files in path
            for root, _, files in os.walk(d_item['path']):
                for f in files:
                    src_file = os.path.join(root, f)

                    # check if filename is regular, matches main source direction and get basename
                    if (not (os.path.isfile(src_file))) :
                        continue
                    if f.endswith(src_suffix):
                        basename = f[:-len(src_suffix)]
                    elif f.endswith(src_suffix + ".gz"):
                        basename = f[:-len(src_suffix) - 3]
                    else:
                        continue

                    # Check all directions are present and aligned
                    # Return opened files and line count
                    files, size = _countLines(os.path.join(root,basename))
                    # Returns 0 if all files do not exist, cannot be aligned or empty
                    if (size == 0) :
                        for f in files :
                            if not f.closed:
                                f.close()
                        continue

                    # build file structure
                    allfiles.append(SamplerFile(root, basename, files, size))

                    # loop over patterns in distribution, check patterns are ok and file matches one
                    for rule in distribution:
                        # distribution is a list of [pattern, weight, addtl options]
                        if (len(rule) < 2 or not isinstance(rule[0], six.string_types)) :
                            raise ValueError('invalid distribution element : %s' % rule)
                        pattern = rule[0]

                        weight = rule[1]
                        extra = None
                        if isinstance(weight, six.string_types) and not weight.startswith('*'):
                            weight = float(weight)
                        if len(rule) > 2:
                            extra = rule[2]
                        if pattern == '*' or re.search(pattern, basename):
                            d_idx_pattern = str(d_idx) + "-" + pattern
                            w = {"pattern": d_idx_pattern, "weight": weight, "extra": extra}
                            allfiles[-1]._weight = w
                            if not isinstance(weight, six.string_types):
                                if d_idx_pattern not in pattern_sizes:
                                    pattern_weights_sum += float(weight)
                                    pattern_sizes[d_idx_pattern] = size
                                else:
                                    pattern_sizes[d_idx_pattern] += size
                            break

        return allfiles, pattern_weights_sum, pattern_sizes


    def _selectLines(f):

        sample_unique = True if 'data' not in config or 'sample_unique' not in config['data'] \
                        else config['data']['sample_unique']

        random_sample = {}

        # Unique sampling, duplicates only if oversampling.
        if not gsample or sample_unique:
            # Minimal number of occurences for each line.
                # 1  if full sample (linekept == linecount or no gsample)
                # >1 if oversampling (linekept > linecount)
                # 0  if undersampling (linekept < linecount)
            min_occurrence = not gsample or int(f._linekept/f._linecount)

            if min_occurrence:
                random_sample = {i:min_occurrence for i in range(f._linecount)}

            # Randomly sampled additional occurences.
            if gsample:
                # Robert Floyd's algorithm for sampling without replacement.
                sampling_size = int(f._linekept - min_occurrence * f._linecount)
                for d in range (f._linecount - sampling_size, f._linecount):
                    t = random.randint(0, d)
                    if t not in random_sample or random_sample[t] == min_occurrence:
                        random_sample[t] = random_sample.get(t, 0) + 1
                    else :
                        random_sample[d] = random_sample.get(d, 0) + 1

        # Simple random sampling, possibly with duplicates.
        else:
            for _ in range(f._linekept):
                i = random.randint(0, f._linecount - 1)
                random_sample[i] = random_sample.get(i, 0) + 1

        f._random_sample = random_sample


    gsample = 0
    if 'data' in config and 'sample' in config['data'] :
        gsample = config['data']['sample']
    else :
        logger.warning('No \'sample\' size specified in configuration,'
                       'all data will be sampled.')

    # TODO multiple sources and targets
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
    allfiles, pattern_weights_sum, pattern_sizes = _discover_files()

    # In strict mode, check that all patterns have been triggered
    if 'data' in config and 'mode_strict' in config['data'] and config['data']['mode_strict']:
        for d_idx, d_item in enumerate(sample_dist):
            for rule in d['distribution'] :
                pattern = rule[0]
                d_idx_pattern = str(d_idx) + "-" + pattern
                if (d_idx_pattern not in pattern_sizes) :
                    raise RuntimeError('pattern %s in block %d doesn\'t match any file with strict mode enabled.' % pattern, d_idx)


    # Adjust weights based on all sampled files for each pattern.
    weights_sum = 0
    weights_size = 0
    reserved_sample = 0
    basenames = set()
    for f in allfiles:
        if hasattr(f, "_weight") and f._weight is not None:
            if f._basename in basenames:
                # Different paths in distribution produced files with the same name.
                # This is not allowed since we write output files in the same folder.
                raise RuntimeError('Two files with the same name %s where sampled.' % f._basename)
            else:
                basenames.add(f._basename)
            linecount = f._linecount
            pattern = f._weight["pattern"]
            weight = f._weight["weight"]
            if isinstance(weight, six.string_types):
                # Oversampling with "*N"
                m = re.match(r"\*([0-9]*)$", weight)
                if not m :
                    raise RuntimeError('Wrong weight format %s for sample pattern %s.' % (weight, pattern))
                oversample = int(m.groups()[0]) if m.groups()[0] else 1
                reserved_sample += linecount * oversample
            else:
                file_weight = float(linecount) / pattern_sizes[pattern]
                pattern_weight = float(f._weight["weight"]) / pattern_weights_sum
                f._weight["weight"] = file_weight * pattern_weight
                weights_sum += f._weight["weight"]
                weights_size += 1
        else:
            logger.debug('No rules matching %s', f._basename)

    # Calculate the number of lines to keep using weights and linecounts, select lines randomly.
    distribute = max(0, gsample - reserved_sample)
    metadata = {}
    summary = {}
    leftover = 0.0
    for f in allfiles:
        extra, pattern = None, None
        f._linekept = 0
        if hasattr(f, "_weight") and f._weight is not None:
            extra = f._weight["extra"]
            pattern = f._weight["pattern"]
            weight = f._weight["weight"]
            linekept = f._linecount
            if gsample and not isinstance(weight, six.string_types):
                weights_size -= 1
                res = distribute * (weight / weights_sum)
                leftover += res - int(res)
                linekept = int(res)
                if leftover > 1.0 :
                    linekept += 1
                    leftover -= 1.0
                if weights_size == 0 and leftover > 0.5 :
                    linekept += 1
                logger.info('Result %d, leftover %f, linekept %d for file %s', res, leftover, linekept, f._basename)


            f._linekept = linekept
        summary[f._basename] = {
            "linecount" : f._linecount,
            "linesampled" : f._linekept,
            "pattern" : pattern
        }
        metadata[f._basename] = extra

        _selectLines(f)

    return allfiles, summary, metadata
