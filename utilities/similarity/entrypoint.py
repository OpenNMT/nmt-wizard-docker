import os
import six
import re
import time
import random
import tempfile

from nmtwizard.utility import Utility
from nmtwizard.logger import get_logger
from nmtwizard.utility import resolve_environment_variables
from nmtwizard.storage import StorageClient

from similarity import main

logger = get_logger(__name__)

class SimilarityUtility(Utility):

    def __init__(self):
        super(SimilarityUtility, self).__init__()

    @property
    def name(self):
        return "similarity"

    def declare_arguments(self, parser):
        subparsers_similarity = parser.add_subparsers(help='Run type', dest='cmd', metavar='{simtrain,simapply}')
        parser_similarity = subparsers_similarity.add_parser('similarity',
                            help='Train or apply similarity model.')
        parser_similarity.add_argument('-mdir', required=True,
                            help='directory to save/restore models')
        parser_similarity.add_argument('-batch_size', required=False, type=int, default=32,
                            help='number of examples per batch [32]')
        parser_similarity.add_argument('-seed', required=False, type=int, default=1234,
                            help='seed for randomness [1234]')
        parser_similarity.add_argument('-debug', action='store_true',
                            help='debug mode')

        parser_train = subparsers_similarity.add_parser('simtrain', parents=[parser_similarity],
                            add_help=False,
                            help='similarity module train mode')
        parser_train.add_argument('-trn', required=True, help='training data')
        parser_train.add_argument('-seq_size', required=False, type=int, default=50,
                            help='sentences larger than this number of src/tgt words are filtered out [50]')
        parser_train.add_argument('-dev', required=False,
                            help='validation data')
        parser_train.add_argument('-src_tok', required=False,
                            help='if provided, json tokenization options for onmt tokenization, points to vocabulary file')
        parser_train.add_argument('-src_voc', required=False,
                            help='vocabulary of src words (needed to initialize learning)')
        parser_train.add_argument('-tgt_tok', required=False,
                            help='if provided, json tokenization options for onmt tokenization, points to vocabulary file')
        parser_train.add_argument('-tgt_voc', required=False,
                            help='vocabulary of tgt words (needed to initialize learning)')
        parser_train.add_argument('-src_emb', required=False,
                            help='embeddings of src words (needed to initialize learning)')
        parser_train.add_argument('-tgt_emb', required=False,
                            help='embeddings of tgt words (needed to initialize learning)')
        parser_train.add_argument('-src_emb_size', required=False, type=int,
                            help='size of src embeddings if -src_emb not used')
        parser_train.add_argument('-tgt_emb_size', required=False,
                            help='size of tgt embeddings if -tgt_emb not used')
        parser_train.add_argument('-src_lstm_size', required=False, type=int, default=256,
                            help='hidden units for src bi-lstm [256]')
        parser_train.add_argument('-tgt_lstm_size', required=False, type=int, default=256,
                            help='hidden units for tgt bi-lstm [256]')
        parser_train.add_argument('-lr', required=False, type=float, default=1.0,
                            help='initial learning rate [1.0]')
        parser_train.add_argument('-lr_decay', required=False, type=float, default=0.9,
                            help='learning rate decay [0.9]')
        parser_train.add_argument('-lr_method', required=False, default='adagrad',
                            help='GD method either: adam, adagrad, adadelta, sgd, rmsprop [adagrad]')
        parser_train.add_argument('-aggr', required=False, default='lse',
                            help='aggregation operation: sum, max, lse [lse]')
        parser_train.add_argument('-r', required=False, type=float, default=1.0,
                            help='r for lse [1.0]')
        parser_train.add_argument('-dropout', required=False, type=float, default=0.3,
                            help='dropout ratio [0.3]')
        parser_train.add_argument('-mode', required=False, default='alignment',
                            help='mode (alignment, sentence) [alignment]')
        parser_train.add_argument('-max_sents', required=False, type=int, default=0,
                            help='Consider this number of sentences per batch (0 for all) [0]')
        parser_train.add_argument('-n_epochs', required=False, type=int, default=1,
                            help='train for this number of epochs [1]')
        parser_train.add_argument('-report_every', required=False, type=int, default=1000,
                            help='report every this many batches [1000]')

        parser_apply = subparsers_similarity.add_parser('simapply', parents=[parser_similarity],
                            add_help=False,
                            help='similarity module apply mode')
        parser_apply.add_argument('-tst_src', required=True, help='testing data, source')
        parser_apply.add_argument('-tst_tgt', required=True, help='testing data, target')
        parser_apply.add_argument('-epoch', required=False, type=int,
                            help='epoch to use ([mdir]/epoch[epoch] must exist, by default the latest one in mdir)')
        parser_apply.add_argument('-output', required=False, default='-',
                            help='output file [- by default is STDOUT]')
        parser_apply.add_argument('-q', required=False, action='store_true',
                            help='quiet mode, just output similarity score')
        parser_apply.add_argument('-show_matrix', required=False, action='store_true',
                            help='output formatted alignment matrix (mode must be alignment)')
        parser_apply.add_argument('-show_svg', required=False, action='store_true',
                            help='output alignment matrix using svg-like html format (mode must be alignment)')
        parser_apply.add_argument('-show_align', required=False, action='store_true',
                            help='output source/target alignment matrix (mode must be alignment)')
        parser_apply.add_argument('-show_last', required=False, action='store_true',
                            help='output source/target last vectors')
        parser_apply.add_argument('-show_aggr', required=False, action='store_true',
                            help='output source/target aggr vectors')

    def exec_function(self, args):
        new_args = []
        local_output = None

        new_args.append('-mdir')
        local_model_dir = self.convert_to_local_file([args.mdir], is_dir=True)[0]
        new_args.append(local_model_dir)
        new_args.append('-batch_size')
        new_args.append(str(args.batch_size))
        new_args.append('-seed')
        new_args.append(str(args.seed))
        if args.debug:
            new_args.append('-debug')
        if args.cmd == 'simtrain':
            new_args.append('-trn')
            new_args.append(self.convert_to_local_file([args.trn])[0])
            new_args.append('-dev')
            new_args.append(self.convert_to_local_file([args.dev])[0])
            new_args.append('-src_tok')
            new_args.append(self.convert_to_local_file([args.src_tok])[0])
            new_args.append('-src_voc')
            new_args.append(self.convert_to_local_file([args.src_voc])[0])
            new_args.append('-tgt_tok')
            new_args.append(self.convert_to_local_file([args.tgt_tok])[0])
            new_args.append('-tgt_voc')
            new_args.append(self.convert_to_local_file([args.tgt_voc])[0])
            new_args.append('-src_emb')
            new_args.append(self.convert_to_local_file([args.src_emb])[0])
            new_args.append('-tgt_emb')
            new_args.append(self.convert_to_local_file([args.tgt_emb])[0])
            new_args.append('-src_emb_size')
            new_args.append(str(args.src_emb_size))
            new_args.append('-tgt_emb_size')
            new_args.append(str(args.tgt_emb_size))
            new_args.append('-src_lstm_size')
            new_args.append(str(args.src_lstm_size))
            new_args.append('-tgt_lstm_size')
            new_args.append(str(args.tgt_lstm_size))
            new_args.append('-lr')
            new_args.append(str(args.lr))
            new_args.append('-lr_decay')
            new_args.append(str(args.lr_decay))
            new_args.append('-lr_method')
            new_args.append(args.lr_method)
            new_args.append('-aggr')
            new_args.append(args.aggr)
            new_args.append('-r')
            new_args.append(str(args.r))
            new_args.append('-dropout')
            new_args.append(str(args.dropout))
            new_args.append('-mode')
            new_args.append(args.mode)
            new_args.append('-max_sents')
            new_args.append(str(args.max_sents))
            new_args.append('-n_epochs')
            new_args.append(str(args.n_epochs))
            new_args.append('-report_every')
            new_args.append(str(args.report_every))
        if args.cmd == 'simapply':
            local_src_file = self.convert_to_local_file([args.tst_src])[0]
            local_tgt_file = self.convert_to_local_file([args.tst_tgt])[0]
            new_args.append('-tst')
            new_args.append(local_src_file + ',' + local_tgt_file)
            if args.epoch:
                new_args.append('-epoch')
                new_args.append(str(args.epoch))
            new_args.append('-output')
            if args.output == '-':
                new_args.append(args.output)
            else:
                local_output = tempfile.NamedTemporaryFile(delete=False)
                new_args.append(local_output.name)
            if args.q:
                new_args.append('-q')
            if args.show_matrix:
                new_args.append('-show_matrix')
            if args.show_svg:
                new_args.append('-show_svg')
            if args.show_align:
                new_args.append('-show_align')
            if args.show_last:
                new_args.append('-show_last')
            if args.show_aggr:
                new_args.append('-show_aggr')

        logger.info("command line option: %s" % " ".join(new_args))
        main(['similarity.py'] + new_args)

        if local_output is not None:
            self._storage.push(local_output.name, args.output)


if __name__ == '__main__':
    SimilarityUtility().run()
