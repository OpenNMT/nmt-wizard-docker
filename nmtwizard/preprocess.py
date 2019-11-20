"""Functions for corpus preprocessing."""

import os

from nmtwizard.logger import get_logger
from nmtwizard.sampler import sample
from nmtwizard import prepoperator
from nmtwizard import tokenizer

logger = get_logger(__name__)

def _generate_models(config, tok_config, corpus_dir, data_dir, option):

    build_option = "build_" + option

    opt_multi = tok_config.get('multi', {}).get(build_option)
    opt_source = tok_config.get('source', {}).get(build_option)
    opt_target = tok_config.get('target', {}).get(build_option)

    if not opt_multi and not (opt_source and opt_target):
        logger.warning('No \'%s\' option specified, exit without preprocessing.' % build_option)
        return

    if (opt_multi and opt_source) or \
       (opt_multi and opt_target) :
        raise RuntimeError('Cannot specify \'%s\' for both \'multi\' and either \'source\' or \'target\'.' % build_option)

    # Generate preprocessed sentences and feed them to subword learners or to vocabularies.
    generate_preprocessed_data(config, corpus_dir, data_dir, option)


def generate_vocabularies(config, corpus_dir, data_dir):

    if "preprocess" not in config:
        raise RuntimeError('No \'preprocess\' field in configuration, \
                            cannot build the vocabularies.)')

    tok_config = None
    for op in reversed(config["preprocess"]):
        if not "op" in op:
            raise RuntimeError('Every step in \'preprocess\' must have a mandatory \'op\' option.')
        # TODO : as is, we always use the last tokenization for buildvocab
        # Should that be an option ?
        if op["op"] == "tokenization":
            tok_config = op
            break

    if not tok_config:
        raise RuntimeError('No \'tokenization\' operator in preprocess configuration, \
                           cannot build the vocabularies.)')

    if ('source' not in tok_config or 'target' not in tok_config) and \
        'multi' not in tok_config :
        raise RuntimeError('Final \'tokenization\' operator used for building vocabulary \
                            should contain either both \'source\' and \'target\' fields \
                            or \'multi\' field.')

    for side in tok_config:
        if side == "op":
            continue
        if tok_config[side].get('vocabulary', {}):
            raise RuntimeError('Cannot build vocabulary if \'%s\' vocabulary path is already specified.' % side)
        build_vocab = tok_config[side].get('build_vocabulary')
        if build_vocab and not build_vocab.get('size'):
            raise RuntimeError('\'size\' option is mandatory to build vocabulary for \'%s\'.' % side)

    _generate_models(config, tok_config, corpus_dir, data_dir, 'subword')

    _generate_models(config, tok_config, corpus_dir, data_dir, 'vocabulary')

    # TODO V2 : we don't need to return tokenization ?
    return {}, tok_config


def generate_preprocessed_data(config, corpus_dir, data_dir, result='preprocess'):

    # TODO V2 : annotations
    # TODO V2 : file-specific rules/extra

    # For backward compatibility with old relative path configurations.
    train_dir = 'train'
    if 'data' in config :
        if 'train_dir' in config['data']:
            train_dir = config['data']['train_dir']
    else :
        logger.warning('No \'data\' field in configuration, \
                        default corpus directory and all corpora are used.)')

    # Default data path.
    data_path = os.path.join(corpus_dir, train_dir)

    num_samples = None
    summary = None
    metadata = None

    # If some sampling OR preprocessing is applied, change result directory.
    if 'data' in config or 'preprocess' in config:

        result_dir = os.path.join(data_dir, result)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        if not os.path.isdir(result_dir):
            raise RuntimeError('%s is not a directory' % result_dir)
        logger.info('Generating data to %s', result_dir)

        # Sample files and write information to a special file structure.
        all_files, summary, metadata = sample(config, data_path)

        num_samples = 0
        consumer = prepoperator.make_consumer(config, result_dir, result)
        for f in all_files:
            lines_filtered = 0
            if f.lines_kept :

                # Default batch size is the whole sample size.
                batch_size = f.lines_kept
                if 'preprocess' in config and 'batch_size' in config['preprocess'] :
                    batch_size = config['preprocess']['batch_size']

                # Loader : load selected lines into batches.
                loader = prepoperator.FileLoader(f, batch_size)

                # Preprocessor : preprocess lines in batch.
                pipeline = prepoperator.PreprocessingPipeline(config)

                # Consumer : action after preprocessing.
                # * write lines to file, if simple preprocessing.
                # * feed to subword learner, if building subword model.
                # * add words to vocabulary, if building vocabulary.
                consumer.open_files(f)

                for tu_batch in loader():
                    tu_batch = pipeline(tu_batch)
                    consumer(tu_batch)
                    lines_filtered += len(tu_batch)
                    # TODO V2 : parallelization
                f.close_files()

                consumer.close_files()

            consumer.finalize(config)

            if lines_filtered != f.lines_kept:
                num_samples += lines_filtered
                summary[f.base_name]["lines_filtered"] = lines_filtered
            else:
                num_samples += f.lines_kept
                summary[f.base_name]["lines_filtered"] = f.lines_kept

        data_path = result_dir

    return data_path, train_dir, num_samples, summary, metadata
