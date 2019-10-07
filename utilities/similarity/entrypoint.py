import os
import six
import re
import time
import random
import tempfile
import subprocess
import shutil
import json

from nmtwizard.utility import Utility
from nmtwizard.logger import get_logger
from nmtwizard.utility import resolve_environment_variables
from nmtwizard.storage import StorageClient

from similarity import main
from build_vocab import main as build_vocab
from build_data import main as build_data
# import pyonmttok after tensorflow
import pyonmttok

logger = get_logger(__name__)

tools_dir = os.getenv('TOOLS_DIR', '/root/tools')

def setCUDA_VISIBLE_DEVICES(gpuid):
    if gpuid == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    else:
        if isinstance(gpuid, list):
            os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(i - 1) for i in gpuid)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid - 1)
        logger.debug(' - set CUDA_VISIBLE_DEVICES= %s' % os.environ['CUDA_VISIBLE_DEVICES'])

def train_tok_model(file_list):
    learner = pyonmttok.SentencePieceLearner(vocab_size=50000, character_coverage=1.0)
    for file in file_list:
        learner.ingest_file(file)

    temp_model_file = tempfile.NamedTemporaryFile(delete=False)
    tokenizer = learner.learn(temp_model_file.name)

    return tokenizer, temp_model_file.name

def train_fast_align(file_src, file_tgt):
    temp_train_file = tempfile.NamedTemporaryFile(delete=False)
    with open(temp_train_file.name, 'w') as outfile:
        with open(file_src) as srcfile, open(file_tgt) as tgtfile:
            for x, y in zip(srcfile, tgtfile):
                x = x.strip()
                y = y.strip()
                outfile.write("{0} ||| {1}\n".format(x, y))

    temp_align_file = tempfile.NamedTemporaryFile(delete=False)
    file_handle = open(temp_align_file.name, "w")
    subprocess.call([os.path.join(tools_dir, 'fast_align'),
                    '-i', temp_train_file.name, '-d', '-o', '-v'],
                    stdout=file_handle)

    os.remove(temp_train_file.name)
    return temp_align_file.name

def build_train_dev_data(file_src, file_tgt, file_align, mode):
    temp_train_file = tempfile.NamedTemporaryFile(delete=False)
    temp_dev_file = tempfile.NamedTemporaryFile(delete=False)

    train_file = open(temp_train_file.name, 'w')
    dev_file = open(temp_dev_file.name, 'w')

    dev_lines = 1000
    n = 0
    outfile = dev_file
    with open(file_src) as srcfile, open(file_tgt) as tgtfile, open(file_align) as alignfile:
        for x, y, a in zip(srcfile, tgtfile, alignfile):
            if(n == dev_lines):
                outfile = train_file
            x = x.strip()
            y = y.strip()
            a = a.strip()
            outfile.write("{0}\t{1}\t{2}\n".format(x, y, a))
            n += 1

    train_file.close()
    dev_file.close()

    temp_train_data_file = tempfile.NamedTemporaryFile(delete=False)
    temp_dev_data_file = tempfile.NamedTemporaryFile(delete=False)

    build_data(['', '-data', temp_train_file.name, '-mode', mode, '-output', temp_train_data_file.name])
    build_data(['', '-data', temp_dev_file.name, '-mode', mode, '-output', temp_dev_data_file.name])

    os.remove(temp_train_file.name)
    os.remove(temp_dev_file.name)
    return temp_train_data_file.name, temp_dev_data_file.name

def generate_tok_config_default(sp_model_filename):
    new_config_file = tempfile.NamedTemporaryFile(delete=False)
    with open(new_config_file.name, 'w') as outfile:
        outfile.write("{\n")
        outfile.write("  \"mode\": \"none\",\n")
        outfile.write("  \"sp_model_path\": \"%s\"\n" % sp_model_filename)
        outfile.write("}\n")

    return new_config_file.name

def tokenize_files(input_file, tokenizer=None, config_file=None):
    if config_file:
        config = None
        with open(config_file) as _config_file:
            config = json.load(_config_file)
        mode = config['mode']
        del config['mode']
        tokenizer = pyonmttok.Tokenizer(mode, **config)

    assert tokenizer is not None
    local_tok_file = tempfile.NamedTemporaryFile(delete=False)
    tokenizer.tokenize_file(input_file, local_tok_file.name)

    return local_tok_file.name

def copy_model_and_update_config(local_model_dir, config_file, source):
    config = None
    with open(config_file) as _config_file:
        config = json.load(_config_file)

    if 'bpe_model_path' in config:
        filename = 'bpe_sim_' + os.path.basename(config['bpe_model_path'])
        shutil.copyfile(config['bpe_model_path'], os.path.join(local_model_dir, filename))
        config['bpe_model_path'] = filename
    if 'sp_model_path' in config:
        filename = 'sp_sim_' + os.path.basename(config['sp_model_path'])
        shutil.copyfile(config['sp_model_path'], os.path.join(local_model_dir, filename))
        config['sp_model_path'] = filename

    config_filename = None
    if source:
        config['vocabulary'] = 'vocab_src'
        config_filename = 'tokenization_src.json'
    else:
        config['vocabulary'] = 'vocab_tgt'
        config_filename = 'tokenization_tgt.json'

    with open(os.path.join(local_model_dir, config_filename), 'w') as outfile:
        json.dump(config, outfile, indent=4)

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
        parser_train.add_argument('-trn_src', required=True, help='training data, source')
        parser_train.add_argument('-trn_tgt', required=True, help='training data, target')
        parser_train.add_argument('-build_data_mode', required=False, default='p',
                            help='how data examples are generated (p: parallel, u:uneven, i:insert, r:replace d:delete) [p]')
        parser_train.add_argument('-seq_size', required=False, type=int, default=50,
                            help='sentences larger than this number of src/tgt words are filtered out [50]')
        parser_train.add_argument('-dev', required=False,
                            help='validation data')
        parser_train.add_argument('-src_tok', required=False,
                            help='if provided, json tokenization options for onmt tokenization source')
        parser_train.add_argument('-tgt_tok', required=False,
                            help='if provided, json tokenization options for onmt tokenization target')
        parser_train.add_argument('-src_tok_sub_model', required=False,
                            help='bpe or SP model for source in src_tok config')
        parser_train.add_argument('-tgt_tok_sub_model', required=False,
                            help='bpe or SP model for target in tgt_tok config')
        parser_train.add_argument('-src_voc', required=False,
                            help='vocabulary of src words (needed to initialize learning)')
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

    def generate_tok_config(self, config_file, sub_tok_model_file):
        local_config_file = self.convert_to_local_file([config_file])[0]
        config = None
        with open(local_config_file) as _config_file:
            config = json.load(_config_file)

        if 'bpe_model' in config:
            config['bpe_model_path'] = config['bpe_model']
            del config['bpe_model']
        if 'bpe_model_path' in config:
            if not sub_tok_model_file:
                raise RuntimeError('find bpe model in config %s, but -[src_tok_sub_model|tgt_tok_sub_model] is not set' % config_file)
            else:
                local_sub_model_file = self.convert_to_local_file([sub_tok_model_file])[0]
                config['bpe_model_path'] = local_sub_model_file

        if 'sp_model' in config:
            config['sp_model_path'] = config['sp_model']
            del config['sp_model']
        if 'sp_model_path' in config:
            if not sub_tok_model_file:
                raise RuntimeError('find SP model in config %s, but -[src_tok_sub_model|tgt_tok_sub_model] is not set' % config_file)
            else:
                local_sub_model_file = self.convert_to_local_file([sub_tok_model_file])[0]
                config['sp_model_path'] = local_sub_model_file

        if 'vocabulary' in config:
            del config['vocabulary']

        new_config_file = tempfile.NamedTemporaryFile(delete=False)
        with open(new_config_file.name, 'w') as outfile:
            json.dump(config, outfile)

        return new_config_file.name

    def prepare_train_data(self, args):
        local_train_src = self.convert_to_local_file([args.trn_src])[0]
        local_train_tgt = self.convert_to_local_file([args.trn_tgt])[0]

        local_train_src_tok = None
        local_train_tgt_tok = None
        local_train_src_config = None
        local_train_tgt_config = None

        if not args.src_tok and not args.tgt_tok:
            logger.info('Start to train joint tokenization model ...')
            tokenizer, local_tok_model = train_tok_model([local_train_src, local_train_tgt])

            local_train_src_config = generate_tok_config_default(local_tok_model)
            logger.info('Start to tokenize source txt ...')
            local_train_src_tok = tokenize_files(local_train_src, tokenizer=tokenizer)

            local_train_tgt_config = generate_tok_config_default(local_tok_model)
            logger.info('Start to tokenize target txt ...')
            local_train_tgt_tok = tokenize_files(local_train_tgt, tokenizer=tokenizer)
        else:
            if args.src_tok:
                local_train_src_config = self.generate_tok_config(args.src_tok, args.src_tok_sub_model)
                logger.info('Start to tokenize source txt ...')
                local_train_src_tok = tokenize_files(local_train_src, config_file=local_train_src_config)
            else:
                logger.info('Start to train source tokenization model ...')
                tokenizer, local_tok_model = train_tok_model([local_train_src])

                local_train_src_config = generate_tok_config_default(local_tok_model)
                logger.info('Start to tokenize source txt ...')
                local_train_src_tok = tokenize_files(local_train_src, tokenizer=tokenizer)
            if args.tgt_tok:
                local_train_tgt_config = self.generate_tok_config(args.tgt_tok, args.tgt_tok_sub_model)
                logger.info('Start to tokenize target txt ...')
                local_train_tgt_tok = tokenize_files(local_train_tgt, config_file=local_train_tgt_config)
            else:
                logger.info('Start to train target tokenization model ...')
                tokenizer, local_tok_model = train_tok_model([local_train_tgt])

                local_train_tgt_config = generate_tok_config_default(local_tok_model)
                logger.info('Start to tokenize target txt ...')
                local_train_tgt_tok = tokenize_files(local_train_tgt, tokenizer=tokenizer)

        logger.info('Start to build source vocab ...')
        local_train_src_vocab = local_train_src_tok + ".vocab"
        build_vocab(['', local_train_src_tok, local_train_src_vocab, 50000])

        logger.info('Start to build target vocab ...')
        local_train_tgt_vocab = local_train_tgt_tok + ".vocab"
        build_vocab(['', local_train_tgt_tok, local_train_tgt_vocab, 50000])

        logger.info('Start to apply fast_align ...')
        local_train_align = train_fast_align(local_train_src_tok, local_train_tgt_tok)

        logger.info('Start to binarize data ...')
        local_train_data, local_dev_data = build_train_dev_data(local_train_src_tok, local_train_tgt_tok, local_train_align, args.build_data_mode)

        os.remove(local_train_src_tok)
        os.remove(local_train_tgt_tok)
        os.remove(local_train_align)

        return {
                'train': local_train_data,
                'dev': local_dev_data,
                'src_voc': local_train_src_vocab,
                'tgt_voc': local_train_tgt_vocab,
                'src_config': local_train_src_config,
                'tgt_config': local_train_tgt_config
               }

    def exec_function(self, args):
        new_args = []
        local_output = None
        train_data = None
        local_model_dir = None

        if args.cmd == 'simtrain':
            local_model_dir = tempfile.mkdtemp()
        if args.cmd == 'simapply':
            local_model_dir = self.convert_to_local_file([args.mdir], is_dir=True)[0]

        new_args.extend([
            '-mdir',        local_model_dir,
            '-batch_size',  str(args.batch_size),
            '-seed',        str(args.seed)
            ])
        if args.debug:
            new_args.append('-debug')
        if args.cmd == 'simtrain':
            setCUDA_VISIBLE_DEVICES(args.gpuid)
            train_data = self.prepare_train_data(args)
            new_args.extend([
                '-trn',     train_data['train'],
                '-dev',     train_data['dev'],
                '-src_voc', train_data['src_voc'],
                '-tgt_voc', train_data['tgt_voc'],
                '-src_emb_size',    str(args.src_emb_size),
                '-tgt_emb_size',    str(args.tgt_emb_size),
                '-src_lstm_size',   str(args.src_lstm_size),
                '-tgt_lstm_size',   str(args.tgt_lstm_size),
                '-lr',              str(args.lr),
                '-lr_decay',        str(args.lr_decay),
                '-lr_method',       args.lr_method,
                '-aggr',            args.aggr,
                '-r',               str(args.r),
                '-dropout',         str(args.dropout),
                '-mode',            args.mode,
                '-max_sents',       str(args.max_sents),
                '-n_epochs',        str(args.n_epochs),
                '-report_every',    str(args.report_every)
                ])
            if args.src_emb:
                new_args.extend(['-src_emb', self.convert_to_local_file([args.src_emb])[0]])
            if args.tgt_emb:
                new_args.extend(['-tgt_emb', self.convert_to_local_file([args.tgt_emb])[0]])

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

        if args.cmd == 'simtrain':
            copy_model_and_update_config(local_model_dir, train_data['src_config'], source=True)
            copy_model_and_update_config(local_model_dir, train_data['tgt_config'], source=False)

            os.remove(train_data['train'])
            os.remove(train_data['dev'])
            os.remove(train_data['src_voc'])
            os.remove(train_data['tgt_voc'])

            self._storage.push(local_model_dir, args.mdir)
            shutil.rmtree(local_model_dir)

        if args.cmd == 'simapply' and local_output is not None:
            self._storage.push(local_output.name, args.output)

if __name__ == '__main__':
    SimilarityUtility().run()
