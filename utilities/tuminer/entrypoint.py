#!/bin/env python3

# nmt-wizard utility for scoring and mining bitexts (translation units or TUs).
# This file is meant to act as an entrypoint in one of the utility dockers for nmt-wizard.
# The docker image is tagged as opennmt/tumining.
#
# Currently, opennmt/tumining relies on a tool LASER developed by Facebook Research
# (https://github.com/facebookresearch/LASER),
# and some code is derived from the python main run script source/mine_bitexts.py.

import os
import sys
import six
import tempfile
import gzip

import numpy as np
import faiss

from nmtwizard.utility import Utility
from nmtwizard.logger import get_logger
from nmtwizard.utility import resolve_environment_variables

logger = get_logger(__name__)


def setCUDA_VISIBLE_DEVICES(gpuid):
    if gpuid == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    else:
        if isinstance(gpuid, list):
            os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(i - 1) for i in gpuid)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid - 1)
        logger.debug(' - set CUDA_VISIBLE_DEVICES= %s' % os.environ['CUDA_VISIBLE_DEVICES'])


LASER = '/opt/LASER'
os.environ['LASER'] = LASER
sys.path.append(LASER + '/source')
sys.path.append(LASER + '/source/lib')

from text_processing import Token, BPEfastApply
from embed import SentenceEncoder, EncodeFile, EmbedLoad
from mine_bitexts import knn, score, score_candidates

def tok(lang, inputF, outputF, verbose):
    Token(inputF,
          outputF,
          lang=lang,
          romanize=True if lang == 'el' else False,
          lower_case=True,
          gzip=True if inputF.endswith('.gz') else False,
          verbose=verbose,
          over_write=False)

def bpe(bpecodes, inputF, outputF, verbose):
    BPEfastApply(inputF,
                 outputF,
                 bpecodes,
                 verbose=verbose,
                 over_write=False)


def emb(encoder, inputF, outputF, verbose, buffer_size):
    EncodeFile(encoder,
               inputF,
               outputF,
               verbose=verbose,
               over_write=False,
               buffer_size=buffer_size)


def loadEncoder(encoderF, buffer_size, max_tokens, max_sentences=None, cpu=False, stable=False):
    buffer_size = max(buffer_size, 1)
    assert not max_sentences or max_sentences <= buffer_size, '--max-sentences/--batch-size ' \
                                                              'cannot be larger than --buffer-size'

    logger.info(' - Encoder: loading {} - cpu={}'.format(encoderF, cpu))
    return SentenceEncoder(encoderF,
                           max_sentences=max_sentences,
                           max_tokens=max_tokens,
                           sort_kind='mergesort' if stable else 'quicksort',
                           cpu=cpu)


def TokBpeEmb(lang, inputF, tokF, bpeF, embF, bpeCodesF, encoder, buffer_size, max_tokens, verbose, gpuid):
    tok(lang, inputF, tokF, verbose)
    bpe(bpeCodesF, tokF, bpeF, verbose)
    setCUDA_VISIBLE_DEVICES(gpuid)

    if isinstance(encoder, str):
        encoder = loadEncoder(encoder, buffer_size, max_tokens, cpu=(gpuid == 0))

    emb(encoder, bpeF, embF, verbose, buffer_size=buffer_size)


def TextLoadUnify(fname, encoding, unify, verbose):
    if verbose:
        print(' - loading texts {:s}: '.format(fname), end='')

    if fname.endswith('.gz'):
        fin = gzip.open(fname, mode='rt', encoding=encoding, errors='surrogateescape')
    else:
        fin = open(fname, mode='r', encoding=encoding, errors='surrogateescape')

    inds = []
    sents = []
    sent2ind = {}
    n = 0
    nu = 0
    for line in fin:
        new_ind = len(sent2ind)
        inds.append(sent2ind.setdefault(line, new_ind))
        if unify:
            if inds[-1] == new_ind:
                sents.append(line[:-1])
                nu += 1
        else:
            sents.append(line[:-1])
            nu += 1
        n += 1
    if verbose:
        print('{:d} lines, {:d} unique'.format(n, nu))
    del sent2ind
    return inds, sents

def unique_embeddings(emb, ind):
    aux = {j: i for i, j in enumerate(ind)}
    logger.info(' - unify embeddings: {:d} -> {:d}'.format(len(emb), len(aux)))
    return emb[[aux[i] for i in range(len(aux))]]


def scoreBitext(src_inds, trg_inds, x, y, x2y_mean, y2x_mean, outputF, encoding, margin):
    logger.info(' - Scoring parallel data')

    if outputF.endswith('.gz'):
        fout = gzip.open(outputF, mode='wt', encoding=encoding, errors='surrogateescape')
    else:
        fout = open(outputF, mode='w', encoding=encoding, errors='surrogateescape')

    for i, j in zip(src_inds, trg_inds):
        fout.write('{:f}\n'.format(score(x[i], y[j], x2y_mean[i], y2x_mean[j], margin)))
    fout.close()


def mineBitext(src_sents, trg_sents, x, y, x2y_ind, x2y_mean, y2x_ind, y2x_mean,
               outputFSrc, outputFTgt, outputFScore, encoding, margin, retrieval, threshold, verbose):
    logger.info(' - mining for parallel data')
    fwd_scores = score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean, margin, verbose)
    bwd_scores = score_candidates(y, x, y2x_ind, y2x_mean, x2y_mean, margin, verbose)
    fwd_best = x2y_ind[np.arange(x.shape[0]), fwd_scores.argmax(axis=1)]
    bwd_best = y2x_ind[np.arange(y.shape[0]), bwd_scores.argmax(axis=1)]

    logger.info(' - writing mined output to {:s}, {:s}, {:s}'.format(outputFSrc, outputFTgt, outputFScore))
    if threshold > 0:
        logger.info(' - with threshold of {:f}'.format(threshold))

    if outputFSrc.endswith('.gz'):
        foutSrc = gzip.open(outputFSrc, mode='wt', encoding=encoding, errors='surrogateescape')
    else:
        foutSrc =open(outputFSrc, mode='w', encoding=encoding, errors='surrogateescape')

    if outputFTgt.endswith('.gz'):
        foutTgt = gzip.open(outputFTgt, mode='wt', encoding=encoding, errors='surrogateescape')
    else:
        foutTgt = open(outputFTgt, mode='w', encoding=encoding, errors='surrogateescape')

    if outputFScore.endswith('.gz'):
        foutTgt = gzip.open(outputFScore, mode='wt', encoding=encoding, errors='surrogateescape')
    else:
        foutScore = open(outputFScore, mode='w', encoding=encoding, errors='surrogateescape')

    def printTriplet(src,tgt,score):
            foutSrc.write('{:s}\n'.format(src))
            foutTgt.write('{:s}\n'.format(tgt))
            foutScore.write('{:f}\n'.format(score))

    if retrieval == 'fwd':
        for i, j in enumerate(fwd_best):
            printTriplet(src_sents[i], trg_sents[j], fwd_scores[i].max())
    if retrieval == 'bwd':
        for j, i in enumerate(bwd_best):
            printTriplet(src_sents[i], trg_sents[j], bwd_scores[j].max())
    if retrieval == 'intersect':
        for i, j in enumerate(fwd_best):
            if bwd_best[j] == i:
                printTriplet(src_sents[i], trg_sents[j], fwd_scores[i].max())
    if retrieval == 'max':
        indices = np.stack((np.concatenate((np.arange(x.shape[0]), bwd_best)),
                            np.concatenate((fwd_best, np.arange(y.shape[0])))), axis=1)
        scores = np.concatenate((fwd_scores.max(axis=1), bwd_scores.max(axis=1)))
        seen_src, seen_trg = set(), set()
        for i in np.argsort(-scores):
            src_ind, trg_ind = indices[i]
            if src_ind not in seen_src and trg_ind not in seen_trg:
                seen_src.add(src_ind)
                seen_trg.add(trg_ind)
                if scores[i] > threshold:
                    printTriplet(src_sents[src_ind], trg_sents[trg_ind], scores[i])
    foutSrc.close()
    foutTgt.close()
    foutScore.close()


def inferLangFromFilename(filename):
    if filename.endswith('.gz'):
        return filename[-5:-3]
    return filename[-2:]


class TuminerUtility(Utility):
    def __init__(self):
        super(TuminerUtility, self).__init__()

    @property
    def name(self):
        return "tuminer"

    def declare_arguments(self, parser):
        parser.add_argument('--mode', required=True, choices=['score', 'mine'],
                            help='Tuminer mode')
        parser.add_argument('--srclang', required=False,
                            help='Source language (two-letter language code; ISO 639-1).')
        parser.add_argument('--srcfile', required=True,
                            help='Source language file.')
        parser.add_argument('--tgtlang', required=False,
                            help='Target language (two-letter language code; ISO 639-1).')
        parser.add_argument('--tgtfile', required=True,
                            help='Target language file.')
        parser.add_argument('--output', required=True,
                            help='Output file.')
        parser.add_argument('--encoding', required=False, default='utf-8',
                            help='Encoding of the input and output text files.')
        parser.add_argument('--verbose', action="store_true",
                            help="Increase output verbosity.")
        parser.add_argument('--threshold', required=False, type=float, default=0,
                            help='When in `mine` mode, threshold value for mined TUs')
        parser.add_argument('--bpecodes', required=False, default=None,
                            help='BPE code to be applied to both source and target files.'
                                 ' (default model provided in docker)')
        parser.add_argument('--encoder', required=False, default=None,
                            help='Multi-lingual encoder to be used to encode both source and target files.'
                                 ' (default model provided in docker)')
        parser.add_argument('--encoderdim', required=False, default=1024,
                            help='Encoder output dimension')
        parser.add_argument('--encoderbuffersize', required=False, type=int, default=10000,
                            help='Encoder buffer size')
        parser.add_argument('--encodermaxtokens', required=False, type=int, default=12000,
                            help='Encoder max_token size')

    def exec_function(self, args):

        setCUDA_VISIBLE_DEVICES(args.gpuid)

        bpeCodesF_local = LASER + '/models/93langs.fcodes'
        encoderF_local = LASER + '/models/bilstm.93langs.2018-12-26.pt'

        srcF_local, tgtF_local, outputF_local = None, None, None

        #################
        # Parse arguments and retrieve files
        #################

        srcF_local = os.path.join(self._data_dir, self._storage.split(args.srcfile)[-1])
        tgtF_local = os.path.join(self._data_dir, self._storage.split(args.tgtfile)[-1])
        self._storage.get_file(args.srcfile, srcF_local)
        self._storage.get_file(args.tgtfile, tgtF_local)

        outputF_local = os.path.join(self._data_dir, self._storage.split(args.output)[-1])

        if args.bpecodes is not None:
            bpeCodesF_local = os.path.join(self._data_dir, self._storage.split(args.bpecodes)[-1])
            self._storage.get_file(args.bpecodes, bpeCodesF_local)
        if args.encoder is not None:
            encoderF_local = os.path.join(self._data_dir, self._storage.split(args.encoder)[-1])
            self._storage.get_file(args.encoder, encoderF_local)

        if args.srclang is None:
            args.srclang = inferLangFromFilename(args.srcfile)
        if args.tgtlang is None:
            args.tgtlang = inferLangFromFilename(args.tgtfile)

        logger.info("srclang: %s, srcfile: %s (%s)" % (args.srclang, args.srcfile, srcF_local))
        logger.info("tgtlang: %s, tgtfile: %s (%s)" % (args.tgtlang, args.tgtfile, tgtF_local))
        logger.info("output: %s (%s)" % (args.output, outputF_local))
        logger.info("encoderF: %s (%s)" % (args.encoder, encoderF_local))
        logger.info("bpeCodesF: %s (%s)" % (args.bpecodes, bpeCodesF_local))

        #################
        # Perform tasks
        #################
        with tempfile.TemporaryDirectory() as tmpdir:
            srcTokF = os.path.join(tmpdir, 'srctok')
            srcBpeF = os.path.join(tmpdir, 'srcbpe')
            srcEmbF = os.path.join(tmpdir, 'srcemb')

            tgtTokF = os.path.join(tmpdir, 'tgttok')
            tgtBpeF = os.path.join(tmpdir, 'tgtbpe')
            tgtEmbF = os.path.join(tmpdir, 'tgtemb')

            logger.debug(' - gpuid: %s' % args.gpuid)

            if isinstance(args.gpuid, list):
                logger.debug(' - perform src and tgt embedding in parallel')

                import torch.multiprocessing as mp

                srcP = mp.Process(target=TokBpeEmb, args=(args.srclang, srcF_local, srcTokF, srcBpeF, srcEmbF,
                                  bpeCodesF_local, encoderF_local, args.encoderbuffersize, args.encodermaxtokens,
                                  args.verbose, args.gpuid[0]))
                srcP.start()

                tgtP = mp.Process(target=TokBpeEmb, args=(args.tgtlang, tgtF_local, tgtTokF, tgtBpeF, tgtEmbF,
                                  bpeCodesF_local, encoderF_local, args.encoderbuffersize, args.encodermaxtokens,
                                  args.verbose, args.gpuid[1]))
                tgtP.start()

                srcP.join()
                tgtP.join()

            else:
                logger.info(' - perform src and tgt embedding in series')
                encoder = loadEncoder(encoderF_local, args.encoderbuffersize, args.encodermaxtokens,
                                      cpu=(args.gpuid == 0))
                TokBpeEmb(args.srclang, srcF_local, srcTokF, srcBpeF, srcEmbF, bpeCodesF_local, encoder,
                          args.encoderbuffersize, args.encodermaxtokens, args.verbose, args.gpuid)
                TokBpeEmb(args.tgtlang, tgtF_local, tgtTokF, tgtBpeF, tgtEmbF, bpeCodesF_local, encoder,
                          args.encoderbuffersize, args.encodermaxtokens, args.verbose, args.gpuid)

            # LASER options
            setCUDA_VISIBLE_DEVICES(args.gpuid)
            unify, retrieval, margin, neighborhood, gpu = True, 'max', 'ratio', 5, (args.gpuid != 0)

            # load bitext
            src_inds, src_sents = TextLoadUnify(srcF_local, args.encoding, unify, args.verbose)
            trg_inds, trg_sents = TextLoadUnify(tgtF_local, args.encoding, unify, args.verbose)

            # load the embeddings
            x = EmbedLoad(srcEmbF, args.encoderdim, verbose=args.verbose)
            if unify:
                x = unique_embeddings(x, src_inds)
            faiss.normalize_L2(x)
            y = EmbedLoad(tgtEmbF, args.encoderdim, verbose=args.verbose)
            if unify:
                y = unique_embeddings(y, trg_inds)
            faiss.normalize_L2(y)

            # calculate knn in both directions
            if retrieval != 'bwd':
                logger.info(' - perform {:d}-nn source against target'.format(neighborhood))
                x2y_sim, x2y_ind = knn(x, y, min(y.shape[0], neighborhood), gpu)
                x2y_mean = x2y_sim.mean(axis=1)

            if retrieval != 'fwd':
                logger.info(' - perform {:d}-nn target against source'.format(neighborhood))
                y2x_sim, y2x_ind = knn(y, x, min(x.shape[0], neighborhood), gpu)
                y2x_mean = y2x_sim.mean(axis=1)

            # margin function
            if margin == 'absolute':
                margin = lambda a, b: a
            elif margin == 'distance':
                margin = lambda a, b: a - b
            else:
                # args.margin == 'ratio':
                margin = lambda a, b: a / b

            if args.mode == 'score':
                scoreBitext(src_inds, trg_inds, x, y, x2y_mean, y2x_mean, outputF_local, args.encoding, margin)
                self._storage.push(outputF_local, args.output)

            elif args.mode == 'mine':
                foutSrc = outputF_local+'.'+args.srclang
                foutSrc_remote = args.output+'.'+args.srclang
                if srcF_local.endswith('.gz'):
                    foutSrc = foutSrc+'.gz'
                    foutSrc_remote = foutSrc_remote+'.gz'
                foutTgt = outputF_local+'.'+args.tgtlang
                foutTgt_remote = args.output+'.'+args.tgtlang
                if tgtF_local.endswith('.gz'):
                    foutTgt = foutTgt+'.gz'
                    foutTgt_remote = foutTgt_remote+'.gz'
                foutScore = outputF_local+'.tuminer-score'
                foutScore_remote = args.output+'.tuminer-score'

                mineBitext(src_sents, trg_sents, x, y, x2y_ind, x2y_mean, y2x_ind, y2x_mean, foutSrc, foutTgt, foutScore,
                           args.encoding, margin, retrieval, args.threshold, args.verbose)

                self._storage.push(foutSrc, foutSrc_remote)
                self._storage.push(foutTgt, foutTgt_remote)
                self._storage.push(foutScore, foutScore_remote)



if __name__ == '__main__':
    TuminerUtility().run()
