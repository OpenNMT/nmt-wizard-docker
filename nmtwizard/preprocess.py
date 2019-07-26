"""Functions for corpus preprocessing."""

import os

from nmtwizard.logger import get_logger
from nmtwizard.sampler import sample
from nmtwizard import prepoperator

logger = get_logger(__name__)

def generate_preprocessed_data(config, corpus_dir, data_dir):

    # TODO : annotations
    # TODO : file-specific rules/extra

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

    # If some sampling OR preprocessing is applied, change directory.
    if 'data' in config or 'preprocess' in config:

        preprocess_dir = os.path.join(data_dir, 'preprocess')
        if not os.path.exists(preprocess_dir):
            os.mkdir(preprocess_dir)
        if not os.path.isdir(preprocess_dir):
            raise RuntimeError('%s is not a directory' % preprocess_dir)
        logger.info('Generating training data to %s', preprocess_dir)

        # Sample files and write information to a special file structure.
        all_files, summary, metadata = sample(config, data_path)

        num_samples = 0
        for f in all_files:
            lines_filtered = 0
            if f.lines_kept :

                # Default batch size is the whole sample size.
                batch_size = f.lines_kept
                if 'preprocess' in config and 'batch_size' in config['preprocess'] :
                    batch_size = config['preprocess']['batch_size']

                loader = prepoperator.FileLoader(f, batch_size)
                pipeline = prepoperator.PreprocessingPipeline()
                # TODO : Initialize FILE-SPECIFIC preprocessor pipeline
                # if 'preprocess' in config:
                # pipeline.add(buildPreprocessPipeline(config['preprocess']))
                # TODO : ultimately, tokenization should be part of the preprocess pipeline
                if 'tokenization' in config:
                    pipeline.add(prepoperator.Tokenizer(config['tokenization']))
                writer = prepoperator.FileWriter(f, preprocess_dir)

                for tu_batch in loader():
                    tu_batch = pipeline(tu_batch)
                    writer(tu_batch)
                    lines_filtered += len(tu_batch)
                    # TODO : parallelization
                f.close_files()
                writer.close_files()

            if lines_filtered != f.lines_kept:
                num_samples += lines_filtered
                summary[f.base_name]["lines_filtered"] = lines_filtered
            else:
                num_samples += f.lines_kept
                summary[f.base_name]["lines_filtered"] = f.lines_kept

        data_path = preprocess_dir

    return data_path, train_dir, num_samples, summary, metadata
