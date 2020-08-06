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


def _ingest_tokens(subword_learner, tu_side):
    if not tu_side.tokens:
        return
    for part in tu_side.tokens:
        for token in part:
            # This method ignores annotations and placeholder tokens.
            subword_learner.ingest_token(token)


class SubwordLearner(Consumer):
    """SubwordLearner class stores, learns and writes subword models."""

    def __init__(self, config, result_dir, tok_step):

        self._result_dir = result_dir

        self._subword_info = {}
        self._subword_learners = {}

        self._tok_step = tok_step

        for side in ('multi', 'source', 'target'):
            tokenization_config = config['preprocess'][tok_step].get(side)
            if tokenization_config is None:
                continue
            subword_config = tokenization_config.get('build_subword')
            if subword_config is None:
                continue
            # The subword learner needs to be aware of the tokenizer annotations
            # to properly ignore them.
            learner_info = tokenizer.make_subword_learner(
                subword_config,
                result_dir,
                tokenizer=tokenizer.build_tokenizer(tokenization_config))
            self._subword_info[side] = learner_info
            self._subword_learners[side] = learner_info['learner']

        self._source_learner = self._subword_learners.get('source')
        self._target_learner = self._subword_learners.get('target')
        self._multi_learner = self._subword_learners.get('multi')


    def __call__(self, tu_batch):
        tu_list, _ = tu_batch
        for tu in tu_list :
            if self._source_learner is not None:
                _ingest_tokens(self._source_learner, tu.src_tok)
            if self._target_learner is not None:
                _ingest_tokens(self._target_learner, tu.tgt_tok)
            if self._multi_learner is not None:
                _ingest_tokens(self._multi_learner, tu.src_tok)
                _ingest_tokens(self._multi_learner, tu.tgt_tok)


    def finalize(self, config, summary=None):

        tok_config = config['preprocess'][self._tok_step]

        # Learn subword models and write them to files.
        for side, subword_info in self._subword_info.items():
            name =  tok_config[side]['build_subword']['name'] \
                    if 'name' in tok_config[side]['build_subword'] \
                    else 'model'+str(self._tok_step)

            subword_type = subword_info['subword_type']
            size = subword_info['size']

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

            subword_info['learner'].learn(out_file)


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


    def finalize(self, config, summary=None):

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
            # target should contain as may parts as source
            target = tu.tgt_tok.tokens if tu.tgt_tok else [None for _ in range(len(tu.src_tok.tokens))]
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
            if tgt_detok is not None:
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
        self._tokens_to_add = {'source':set(), 'target':set()}
        self.num_samples = 0

    def open_files(self, f, build_state):
        # TODO V2 : multiple files
        # TODO V2 : do we output ALL the files that we take as input ?
        self._lines_filtered = 0
        self._f = f
        self._files = {}
        src = os.path.join(self._result_dir, f.base_name + "." + f.src_suffix)
        self._files["src"] = open(src, 'w')
        tgt = os.path.join(self._result_dir, f.base_name + "." + f.tgt_suffix)
        self._files["tgt"] = open(tgt, 'w')
        if build_state.get('write_alignment', False):
            align = os.path.join(self._result_dir, f.base_name + ".align")
            self._files["align"] = open(align, 'w')

    def close_files(self):
        for f in self._files.values():
            f.close()

    def __call__(self, tu_batch):
        tu_list, meta = tu_batch
        if 'tokens_to_add' in meta:
            if 'source' in meta['tokens_to_add']:
                self._tokens_to_add['source'].update(meta['tokens_to_add']['source'])
            if 'target' in meta['tokens_to_add']:
                self._tokens_to_add['target'].update(meta['tokens_to_add']['target'])
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


            if "align" in self._files:
                alignment = tu.alignment
                if alignment :
                    for part in alignment:
                        part = " ".join("%s-%s" % tup for tup in part)
                        self._files["align"].write("%s\n" % part)
        self._lines_filtered += len(tu_list)


    def finalize(self, config, summary=None):
        if self._lines_filtered != self._f.lines_kept:
            self.num_samples += self._lines_filtered
            summary[self._f.base_name]["linefiltered"] = self._lines_filtered
        else:
            self.num_samples += self._f.lines_kept
            summary[self._f.base_name]["linefiltered"] = self._f.lines_kept

        if self._tokens_to_add['source'] or self._tokens_to_add['target'] :
            if 'tokens_to_add' not in summary:
                summary['tokens_to_add'] = {}
            if 'source' not in summary['tokens_to_add']:
                summary['tokens_to_add']['source'] = []
            if 'target' not in summary['tokens_to_add']:
                summary['tokens_to_add']['target'] = []
            source_set = set(summary['tokens_to_add']['source'])
            source_set.update(self._tokens_to_add['source'])
            summary['tokens_to_add']['source'] = list(source_set)
            target_set = set(summary['tokens_to_add']['target'])
            target_set.update(self._tokens_to_add['target'])
            summary['tokens_to_add']['target'] = list(target_set)


def make_consumer(config, result_dir, result, tok_step):

    if result == 'subword':
        return SubwordLearner(config, result_dir, tok_step)

    if result == 'vocabulary':
        return VocabularyBuilder(config, result_dir, tok_step)

    # Default is write to file.
    return SamplerFileWriter(result_dir)
