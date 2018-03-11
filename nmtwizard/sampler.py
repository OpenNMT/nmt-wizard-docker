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

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def count_lines(path):
    if os.path.isfile(path + ".gz"):
        f = gzip.open(path + ".gz", 'rb')
    elif not os.path.isfile(path):
        raise RuntimeError('%s is not a file' % path)
    else:
        f = open(path, 'rb')
    i = 0
    for i, _ in enumerate(f):
        pass
    f.close()
    return i + 1

def sample(gsample, sample_dist, source_dir, target_dir, src_suffix, tgt_suffix):

    def _countLine(filepath):
        logger.debug("Processing %s", filepath[0])
        src_lines = count_lines(filepath[0] + src_suffix)
        tgt_lines = count_lines(filepath[0] + tgt_suffix)
        if src_lines != tgt_lines:
            raise RuntimeError('source and target line count mismatch (%d/%d)'
                               % (src_lines, tgt_lines))
        if src_lines == 0:
            raise RuntimeError('file %s is empty' % filepath[0])
        return src_lines

    def _buildFile(fullpath, file, count, totalcount, rule):
        logger.debug("Building %d/%d in %s (rule: %s)", count, totalcount, file, rule)
        src_compressed = os.path.isfile(fullpath + src_suffix + ".gz")
        tgt_compressed = os.path.isfile(fullpath + tgt_suffix + ".gz")
        if count == totalcount and not src_compressed and not tgt_compressed:
            # copy files
            shutil.copyfile(fullpath + src_suffix, os.path.join(target_dir, file) + src_suffix)
            shutil.copyfile(fullpath + tgt_suffix, os.path.join(target_dir, file) + tgt_suffix)
        else:
            # sample
            rand_smpl = sorted(random.randint(0, totalcount - 1) for _ in range(count))
            idx = 0

            if src_compressed:
                fisrc = gzip.open(fullpath + src_suffix + ".gz", 'rb')
            else:
                fisrc = open(fullpath + src_suffix, 'rb')

            if tgt_compressed:
                fitgt = gzip.open(fullpath + tgt_suffix + ".gz", 'rb')
            else:
                fitgt = open(fullpath + tgt_suffix, 'rb')

            fosrc = open(os.path.join(target_dir, file) + src_suffix, "wb")
            fotgt = open(os.path.join(target_dir, file) + tgt_suffix, "wb")
            lidx = 0
            while idx < len(rand_smpl):
                lsrc = fisrc.readline()
                ltgt = fitgt.readline()
                while idx < count and lidx == rand_smpl[idx]:
                    fosrc.write(lsrc)
                    fotgt.write(ltgt)
                    idx += 1
                lidx += 1
            fisrc.close()
            fitgt.close()
            fosrc.close()
            fotgt.close()

    if not os.path.exists(source_dir) or not os.path.isdir(source_dir):
        raise RuntimeError('source directory %s does not exist' % source_dir)

    if not os.path.exists(target_dir):
        mkdir_p(target_dir)

    if not os.path.isdir(target_dir):
        raise RuntimeError('target directory %s does not exist' % target_dir)

    for d in sample_dist:
        if "path" not in d or "distribution" not in d:
            raise ValueError('invalid distribution in collection %s' % d)

    # start collecting the files
    allfiles = []
    for ri in range(len(sample_dist)):
        for root, _, files in os.walk(os.path.join(source_dir, sample_dist[ri]['path'])):
            for file in files:
                if file.endswith(src_suffix):
                    fullpath = os.path.join(root, file[:-len(src_suffix)])
                    allfiles.append((fullpath, file[:-len(src_suffix)], ri))
                elif file.endswith(src_suffix + ".gz"):
                    fullpath = os.path.join(root, file[:-len(src_suffix) - 3])
                    allfiles.append((fullpath, file[:-len(src_suffix) - 3], ri))

    weights = []
    pattern_weights_sum = 0
    pattern_sizes = {}
    file_sizes = []
    for (fp, f, ri) in allfiles:
        size = -1
        w = None
        # distribution is a list of ["pattern", weight, "options", ...]
        for rule in sample_dist[ri]['distribution']:
            pattern = rule[0]
            weight = rule[1]
            extra = rule[2:]
            if weight != '*' and isinstance(weight, six.string_types):
                weight = float(weight)
            if pattern == '*' or re.search(pattern, f):
                size = _countLine((fp, f, ri))
                w = {"pattern": pattern, "weight": weight, "extra": extra}
                if weight != '*':
                    if pattern not in pattern_sizes:
                        pattern_weights_sum += float(weight)
                        pattern_sizes[pattern] = size
                    else:
                        pattern_sizes[pattern] += size
                break
        file_sizes.append(size)
        weights.append(w)

    weights_sum = 0
    reserved_sample = 0
    for (f, c, w) in zip(allfiles, file_sizes, weights):
        if w is not None:
            if w["weight"] == '*':
                reserved_sample += c
            else:
                file_weight = float(c) / pattern_sizes[w["pattern"]]
                pattern_weight = float(w["weight"]) / pattern_weights_sum
                w["weight"] = file_weight * pattern_weight
                weights_sum += w["weight"]

    distribute = max(0, gsample - reserved_sample)
    summary_by_pattern = collections.defaultdict(int)
    summary_by_file = {}
    metadata = {}
    samplefile = []
    for (f, c, w) in zip(allfiles, file_sizes, weights):
        if w is None:
            logger.debug('No rules matching %s', f[0])
        else:
            metadata[f[1]] = w["extra"]
            if w["weight"] == '*':
                samplefile.append((f[0], f[1], c, c, w["pattern"]))
                summary_by_pattern[w["pattern"]] += c
                summary_by_file[f[0]] = c
            else:
                nsent = int(round(distribute * (w["weight"] / weights_sum)))
                summary_by_pattern[w["pattern"]] += nsent
                summary_by_file[f[0]] = nsent
                if nsent >= 1:
                    samplefile.append((f[0], f[1], nsent, c, w["pattern"]))
                else:
                    logger.debug('Zero line to select for: %s', f[0])

    for sample in samplefile:
        _buildFile(*sample)

    return {"pattern": summary_by_pattern, "file": summary_by_file}, metadata

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--sample_dist', required=True,
                        help='sampling distribution file')
    parser.add_argument('-S', '--source_dir', required=True,
                        help='source directory where to sample from')
    parser.add_argument('-T', '--target_dir', required=True,
                        help='target directory where to sample to')
    parser.add_argument('-s', '--src_suffix', required=True,
                        help='suffix of source language files')
    parser.add_argument('-t', '--tgt_suffix', required=True,
                        help='suffix of target language files')
    parser.add_argument('-g', '--gsample', type=int, required=True,
                        help='number of sentences to sample')
    parser.add_argument('-l', '--log_level', default='INFO',
                        help='log level')

    args = parser.parse_args()

    logger.setLevel(args.log_level)

    if not os.path.exists(args.sample_dist) or not os.path.isfile(args.sample_dist):
        raise ValueError('distribution file %s does not exist or is not a file' % args.sample_dist)

    with open(args.sample_dist) as data:
        sample_dist = json.load(data)

    if not isinstance(sample_dist, list):
        raise ValueError('sample_dist should a collection of {path, [sample_rules]}')

    sample(args.gsample, sample_dist,
           args.source_dir, args.target_dir,
           args.src_suffix, args.tgt_suffix)

if __name__ == "__main__":
    main()
