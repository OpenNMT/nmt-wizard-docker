# coding: utf-8
import six
import abc
import os
import collections
import itertools

from nmtwizard.preprocess import tokenizer

@six.add_metaclass(abc.ABCMeta)
class Consumer(object):
    """Base class for using preprocess results."""

    @abc.abstractmethod
    def __call__(self, tu_batch):
        raise NotImplementedError()


class SubwordLearner(Consumer):
    """SubwordLearner class stores, learns and writes subword models."""

    def __init__(self, config, result_dir, tok_step):

        self._result_dir = result_dir

        self._subword_learners = {}

        self._tok_step = tok_step
        tok_config = config['preprocess'][tok_step]

        opt_multi = tok_config.get('multi', {}).get('build_subword')
        opt_source = tok_config.get('source', {}).get('build_subword')
        opt_target = tok_config.get('target', {}).get('build_subword')

        if opt_multi:
            self._subword_learners['multi'] = tokenizer.make_subword_learner(opt_multi, result_dir)
        if opt_source:
            self._subword_learners['source'] = tokenizer.make_subword_learner(opt_source, result_dir)
        if opt_target:
            self._subword_learners['target'] = tokenizer.make_subword_learner(opt_target, result_dir)


    def __call__(self, tu_batch):
        # Feed lines to subword learners.
        # TODO V2 : feed tokenized lines, individual tokens ?
        # TODO V2 : undo all placeholder annotation for subword processing
        tu_list, _ = tu_batch
        for tu in tu_list :
            if 'source' in self._subword_learners:
                self._subword_learners['source']['learner'].ingest(tu.src_detok)
            if 'target' in self._subword_learners:
                self._subword_learners['target']['learner'].ingest(tu.tgt_detok)
            if 'multi' in self._subword_learners:
                self._subword_learners['multi']['learner'].ingest(tu.src_detok)
                self._subword_learners['multi']['learner'].ingest(tu.tgt_detok)


    def finalize(self, config):

        tok_config = config['preprocess'][self._tok_step]

        # Learn subword models and write them to files.
        for side, learner in self._subword_learners.items():
            name =  tok_config[side]['build_subword']['name'] \
                    if 'name' in tok_config[side]['build_subword'] \
                    else 'model'+str(self._tok_step)

            subword_type = self._subword_learners[side]['subword_type']
            size = self._subword_learners[side]['size']

            if side == 'multi' :
                out_file = os.path.join(self._result_dir, \
                                        "joint_%s_%s-%d.%s_%s" % \
                                        (subword_type, name, size, config['source'], config['target']) )
                tok_config['source'][subword_type+"_model_path"] = out_file
                tok_config['target'][subword_type+"_model_path"] = out_file
            else :
                out_file = os.path.join(self._result_dir, \
                                        "%s_%s-%d.%s" % (subword_type, name, size, config[side]))
                tok_config[side][subword_type+"_model_path"] = out_file

            self._subword_learners[side]['learner'].learn(out_file)


class VocabularyBuilder(Consumer):
    """VocabularyBuilder class stores, learns and writes vocabularies."""

    def __init__(self, config, result_dir, tok_step):

        self._result_dir = result_dir

        self._vocabularies = {}
        self._sums = {}

        self._tok_step = tok_step
        tok_config = config['preprocess'][tok_step]

        opt_multi = tok_config.get('multi', {}).get('build_vocabulary')
        opt_source = tok_config.get('source', {}).get('build_vocabulary')
        opt_target = tok_config.get('target', {}).get('build_vocabulary')

        if opt_multi:
            self._vocabularies['multi'] = collections.defaultdict(int)
            self._sums['multi'] = 0
        if opt_source:
            self._vocabularies['source'] = collections.defaultdict(int)
            self._sums['source'] = 0
        if opt_target:
            self._vocabularies['target'] = collections.defaultdict(int)
            self._sums['target'] = 0


    def __call__(self, tu_batch):

        tu_list, _ = tu_batch
        # TODO : remove value for placeholders
        for tu in tu_list :
            if 'source' in self._vocabularies:
                for token in itertools.chain.from_iterable(tu.src_tok.tokens):
                    self._vocabularies['source'][token] += 1
                    self._sums['source'] += 1
            if 'target' in self._vocabularies:
                for token in itertools.chain.from_iterable(tu.tgt_tok.tokens):
                    self._vocabularies['target'][token] += 1
                    self._sums['target'] += 1
            if 'multi' in self._vocabularies:
                for token in itertools.chain.from_iterable(tu.src_tok.tokens):
                    self._vocabularies['multi'][token] += 1
                    self._sums['multi'] += 1
                for token in itertools.chain.from_iterable(tu.tgt_tok.tokens):
                    self._vocabularies['multi'][token] += 1
                    self._sums['multi'] += 1


    def _prune(self, vocabulary, sorted_vocabulary, size, min_frequency):
        real_size = len(sorted_vocabulary)

        if min_frequency :
            for t in reversed(sorted_vocabulary):
                if vocabulary[t] < min_frequency :
                    real_size -= 1
                else:
                    break

        return min(real_size, size)


    def finalize(self, config):

        tok_config = config['preprocess'][self._tok_step]

        for side, vocabulary in self._vocabularies.items():
            name =  tok_config[side]['build_vocabulary']['name'] \
                    if 'name' in tok_config[side]['build_vocabulary'] \
                    else 'vocab'+str(self._tok_step)

            # Size option is mandatory, already checked it.
            size = tok_config[side]['build_vocabulary']['size']

            min_frequency = tok_config[side]['build_vocabulary']['min-frequency'] \
                            if 'min-frequency' in tok_config[side]['build_vocabulary'] \
                            else 0

            added_size = 0

            # Merge previously created vocabulary.
            vocab_to_merge = tok_config[side]['build_vocabulary']['merge'] \
                             if 'merge' in tok_config[side]['build_vocabulary'] \
                             else None

            if vocab_to_merge and os.path.isfile(vocab_to_merge):
                with open(vocab_to_merge, 'r') as f:
                    header = True
                    for l in f:
                        if header and l[0] == '#':
                            continue
                        header = False
                        w = l.strip().split(' ')[0]
                        if w :
                            # Set heaviest frequency on tokens from vocabulary to merge.
                            vocabulary[w] = float("inf")
                            added_size += 1

            # Add extra tokens from a list.
            vocab_to_add = tok_config[side]['build_vocabulary']['add'] \
                           if 'add' in tok_config[side]['build_vocabulary'] \
                           else []

            for w in vocab_to_add:
                vocabulary[w] = float("inf")
                added_size += 1

            if added_size > size :
                raise RuntimeError('The size of extra tokens from \'merge\' and \'add\' (%d) cannot be bigger than than the required vocabulary size (%d)' % (added_size, size))

            # Find out the real vocabulary size.
            sorted_vocabulary = sorted(vocabulary, key=vocabulary.get, reverse=True)

            real_size = self._prune(vocabulary, sorted_vocabulary, size, min_frequency)

            # Write to file.
            if side == 'multi' :
                out_file = os.path.join(self._result_dir, \
                                        "joint_%s-%d.%s_%s" % \
                                        (name, real_size, config['source'], config['target']))
                tok_config['source']['vocabulary_path'] = out_file
                tok_config['target']['vocabulary_path'] = out_file

            else :
                out_file = os.path.join(self._result_dir, "%s-%d.%s" % (name, real_size, config[side]))
                tok_config[side]['vocabulary_path'] = out_file

            with open(out_file, 'w') as vocab_file :
                for i in range(real_size):
                    w = sorted_vocabulary[i]
                    vocab_file.write("%s %s\n" % (w, vocabulary[w]/float(self._sums[side])))

            # TODO V2 : header with configuration ?
            # TODO V2 : deal with placeholders

class BasicWriter(Consumer):
    """BasicWriter writes one pre/postprocessed TU at inference."""

    def __init__(self, postprocess):
        self._postprocess = postprocess

    def __call__(self, tu_batch):
        """ In preprocess, output is ((source, metadata), target), tokenized and possibly multipart, where target is either None or incomplete translation.

            In postprocess, output is postprocessed target, untokenized and one-part."""

        tu_list, _ = tu_batch
        tu = tu_list[0]
        # Postprocess.
        if self._postprocess:
            self.output = tu.tgt_detok
        # Preprocess in inference.
        else:
            target = tu.tgt_tok.tokens if tu.tgt_tok else [None]
            self.output = ((tu.src_tok.tokens, tu.metadata), target)


class FileWriter(Consumer):
    """FileWriter writes pre/postprocessed TUs into files at inference."""

    def __init__(self, output_file):
        # A basic file writer only opens one file.
        # In preprocess, it is used to store preprocessed source.
        # In postprocess, it is used to store postprocessed target.
        # TODO V2 : multiple files
        self._file = open(output_file, 'w')
        self.metadata = []


    def close_files(self):
        self._file.close()


    def __call__(self, tu_batch):
        tu_list, _ = tu_batch
        # Write lines to files from TUs
        for tu in tu_list :
            tgt_detok = tu.tgt_detok
            # Postprocess.
            if tgt_detok:
                self._file.write("%s\n" % tgt_detok)
            # Preprocess.
            else:
                for part in tu.src_tok.tokens:
                    part = " ".join(part)
                    self._file.write("%s\n" % part)
                self.metadata.append(tu.metadata)


class SamplerFileWriter(Consumer):
    """SamplerFileWriter writes pre/postprocessed TUs into files at training using SamplerFile object."""

    def __init__(self, result_dir):
        self._result_dir = result_dir

    def open_files(self, f):
        # TODO V2 : multiple files
        # TODO V2 : do we output ALL the files that we take as input ?
        self._files = {}
        src = os.path.join(self._result_dir, os.path.basename(f.files["src"].name))
        self._files["src"] = open(src, 'w')
        tgt = os.path.join(self._result_dir, os.path.basename(f.files["tgt"].name))
        self._files["tgt"] = open(tgt, 'w')

    def close_files(self):
        for f in self._files.values():
            f.close()

    def __call__(self, tu_batch):
        tu_list, _ = tu_batch
        # Write lines to file from TUs
        for tu in tu_list :
            src_tokens = tu.src_tok.tokens
            if src_tokens :
                for part in src_tokens :
                    part = " ".join(part)
                    self._files["src"].write("%s\n" % part)
            else:
                self._files["src"].write("%s\n" % tu.src_detok)

            tgt_tokens = tu.tgt_tok.tokens
            if tgt_tokens :
                for part in tgt_tokens :
                    part = " ".join(part)
                    self._files["tgt"].write("%s\n" % part)
            else :
                self._files["tgt"].write("%s\n" % tu.tgt_detok)

def make_consumer(config, result_dir, result, tok_step):

    if result == 'subword':
        return SubwordLearner(config, result_dir, tok_step)

    if result == 'vocabulary':
        return VocabularyBuilder(config, result_dir, tok_step)

    # Default is write to file.
    return SamplerFileWriter(result_dir)
