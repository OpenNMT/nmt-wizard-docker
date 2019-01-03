import os
import six
import re
import time
import random
import argparse
import subprocess

from nmtwizard.utility import Utility
from nmtwizard.logger import get_logger
from nmtwizard.utility import resolve_environment_variables
from nmtwizard import data

logger = get_logger(__name__)

class WordAlignUtility(Utility):

    def __init__(self):
        super(WordAlignUtility, self).__init__()

        subparsers = self._parser.add_subparsers(help='Run type', dest='cmd')
        parser_train = subparsers.add_parser('train', help='Run a word align training.')

        parser_train.add_argument('--input', '-i', help='Input files (comma-separated, only useful if'
                                                        ' model is not provided)')
        parser_train.add_argument('--output_dir', '-O', help='output directory', required=True)
        parser_train.add_argument('--conditional_probability_filename', '-p', type=str)
        parser_train.add_argument('--reverse', '-r', default=False, action='store_true',
                                  help='Run alignment in reverse (condition on target and predict source)')
        parser_train.add_argument('--favor_diagonal', '-d', default=False, action='store_true',
                                  help='Favor alignment points close to the monotonic diagonoal')
        parser_train.add_argument('--force_align', '-f', type=str)
        parser_train.add_argument('--iterations', '-I', type=int, default=5,
                                  help='number of iterations in EM training')
        parser_train.add_argument('--mean_srclen_multiplier', '-m', type=float, default=1.0)
        parser_train.add_argument('--beam_threshold', '-t', type=float, default=-4.0)
        parser_train.add_argument('--prob_align_null', '-q', type=float, default=0.08,
                                  help='p_null parameter')
        parser_train.add_argument('--diagonal_tension', '-T', type=float, default=4.0,
                                  help='starting lambda for diagonal distance parameter')
        parser_train.add_argument('--variational_bayes', '-v', default=False, action='store_true',
                                  help='Use Dirichlet prior on lexical translation distributions')
        parser_train.add_argument('--optimize_tension', '-o', type=float, default=0.01,
                                  help='alpha parameter for optional Dirichlet prior')
        parser_train.add_argument('--no_null_word', '-N', default=False, action='store_true',
                                  help='No null word')
        parser_train.add_argument('--thread_buffer_size', '-b', type=int, default=10000)
        parser_train.add_argument('--print_scores', '-s', default=False, action='store_true',
                                  help='print alignment scores')

    @property
    def name(self):
        return "wordalign"

    def exec_function(self, args):
        parser = self._parser
        args = parser.parse_args(args=args)
        if self._model is None and args.input is None:
            parser.error('wordalign train requires preprocess model or input files')
        if self._model is not None and args.input is not None:
            parser.error('wordalign train cannot use both preprocess model and input files')
        if args.conditional_probability_filename is not None:
            parser.error('wordalign train requires -o option, and not -p')
        if args.reverse:
            parser.error('wordalign train should not take -r option')
        output_dir = resolve_environment_variables(args.output_dir)
        if os.path.exists(output_dir):
            if not os.path.isdir(output_dir):
                parser.error('-o should be a directory')
        else:
            os.makedirs(output_dir)

        if self._model is not None:
            remote_model_path = self._storage.join(self._model_storage_read, self._model)
            model_path = os.path.join(self._models_dir, self._model)
            fetch_model(self._storage, remote_model_path, model_path)
            with open(os.path.join(model_path, 'config.json'), 'r') as config_file:
                model_config = json.load(config_file)
            if model_config.get('modelType') != 'preprocess':
                raise ValueError('wordalign training require preprocess model to train')

            data_dir = self._merge_multi_training_files(
                data_dir, train_dir, model_config['source'], model_config['target'])
            input_files = (os.path.join(data_dir, 'train.%s' % model_config['source']),
                           os.path.join(data_dir, 'train.%s' % model_config['target']))
        else:
            input_files = resolve_environment_variables(args.input.split(','))
            if len(input_files) > 2:
                parser.error('maximum 2 input files are required')

        if not os.path.exists(input_files[0]):
            parser.error('cannot find file: %s' % input_files[0])
        if len(input_files) == 2:
            if not os.path.exists(input_files[1]):
                parser.error('cannot find file: %s' % input_files[1])
            output_file = os.path.join(self._data_dir, 'pasted')
            data.paste_files(input_files, output_file, ' ||| ')
            input_files = (output_file,)
        else:
            logger.warning('single file provided, assuming fast_align input format')

        # fast_align input is space-tokenized file with ||| separator
        cmd = ['fast_align', '-i', input_files[0]]
        cmd += ['-p', os.path.join(output_dir, 'forward.probs')]
        if args.favor_diagonal:
            cmd += ['-d']
        if args.force_align:
            cmd += ['-f', resolve_environment_variables(args.force_align)]
        if args.iterations:
            cmd += ['-I', str(args.iterations)]
        if args.mean_srclen_multiplier:
            cmd += ['-m', str(args.mean_srclen_multiplier)]
        if args.beam_threshold:
            cmd += ['-t', str(args.beam_threshold)]
        if args.prob_align_null:
            cmd += ['-q', str(args.prob_align_null)]
        if args.diagonal_tension:
            cmd += ['-T', str(args.diagonal_tension)]
        if args.optimize_tension:
            cmd += ['-o', str(args.optimize_tension)]
        if args.variational_bayes:
            cmd += ['-v']
        if args.no_null_word:
            cmd += ['-N']
        if args.print_scores:
            cmd += ['-s']
        if args.thread_buffer_size:
            cmd += ['-b', str(args.thread_buffer_size)]

        logger.info("calling: %s", " ".join(cmd))
        subprocess.call(cmd)

        cmd[4] = os.path.join(output_dir, 'backward.probs')
        cmd += ['-r']

        logger.info("calling: %s", " ".join(cmd))
        subprocess.call(cmd)


if __name__ == '__main__':
    WordAlignUtility().run()
