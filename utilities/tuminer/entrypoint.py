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
from mine_bitexts import TextLoadUnify, knn, score, score_candidates

def tok(lang, inputF, outputF, verbose):
    Token(inputF,
          outputF,
          lang=lang,
          romanize=True if lang == 'el' else False,
          lower_case=True,
          gzip=False,
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


def unique_embeddings(emb, ind):
    aux = {j: i for i, j in enumerate(ind)}
    logger.info(' - unify embeddings: {:d} -> {:d}'.format(len(emb), len(aux)))
    return emb[[aux[i] for i in range(len(aux))]]


def scoreBitext(src_inds, trg_inds, x, y, x2y_mean, y2x_mean, outputF, encoding, margin):
    logger.info(' - Scoring parallel data')
    fout = open(outputF, mode='w', encoding=encoding, errors='surrogateescape')
    for i, j in zip(src_inds, trg_inds):
        print(score(x[i], y[j], x2y_mean[i], y2x_mean[j], margin), file=fout)
    fout.close()


def mineBitext(src_sents, trg_sents, x, y, x2y_ind, x2y_mean, y2x_ind, y2x_mean, outputF, encoding, margin,
               retrieval, threshold, verbose):
    logger.info(' - mining for parallel data')
    fwd_scores = score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean, margin, verbose)
    bwd_scores = score_candidates(y, x, y2x_ind, y2x_mean, x2y_mean, margin, verbose)
    fwd_best = x2y_ind[np.arange(x.shape[0]), fwd_scores.argmax(axis=1)]
    bwd_best = y2x_ind[np.arange(y.shape[0]), bwd_scores.argmax(axis=1)]

    logger.info(' - writing alignments to {:s}'.format(outputF))
    if threshold > 0:
        logger.info(' - with threshold of {:f}'.format(threshold))

    fout = open(outputF, mode='w', encoding=encoding, errors='surrogateescape')

    if retrieval == 'fwd':
        for i, j in enumerate(fwd_best):
            print(fwd_scores[i].max(), src_sents[i], trg_sents[j], sep='\t', file=fout)
    if retrieval == 'bwd':
        for j, i in enumerate(bwd_best):
            print(bwd_scores[j].max(), src_sents[i], trg_sents[j], sep='\t', file=fout)
    if retrieval == 'intersect':
        for i, j in enumerate(fwd_best):
            if bwd_best[j] == i:
                print(fwd_scores[i].max(), src_sents[i], trg_sents[j], sep='\t', file=fout)
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
                    print(scores[i], src_sents[src_ind], trg_sents[trg_ind], sep='\t', file=fout)
    fout.close()


class TuminerUtility(Utility):
    def __init__(self):
        super(TuminerUtility, self).__init__()

    @property
    def name(self):
        return "tuminer"

    def declare_arguments(self, parser):
        parser.add_argument('--mode', required=True, choices=['score', 'mine'],
                            help='Tuminer mode')
        parser.add_argument('--srclang', required=True,
                            help='Source language (two-letter language code; ISO 639-1).')
        parser.add_argument('--srcfile', required=True,
                            help='Source language file.')
        parser.add_argument('--tgtlang', required=True,
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

        if args.output != '-':
            outputF_local = os.path.join(self._data_dir, self._storage.split(args.output)[-1])

        if args.bpecodes is not None:
            bpeCodesF_local = os.path.join(self._data_dir, self._storage.split(args.bpecodes)[-1])
            self._storage.get_file(args.bpecodes, bpeCodesF_local)
        if args.encoder is not None:
            encoderF_local = os.path.join(self._data_dir, self._storage.split(args.encoder)[-1])
            self._storage.get_file(args.encoder, encoderF_local)

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

            args.unify = unify

            # load bitext
            src_inds, src_sents = TextLoadUnify(srcF_local, args)
            trg_inds, trg_sents = TextLoadUnify(tgtF_local, args)

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
            elif args.mode == 'mine':
                mineBitext(src_sents, trg_sents, x, y, x2y_ind, x2y_mean, y2x_ind, y2x_mean, outputF_local,
                           args.encoding, margin, retrieval, args.threshold, args.verbose)

        #################
        # Save output
        #################
        if outputF_local is not None:
            self._storage.push(outputF_local, args.output)


if __name__ == '__main__':
    TuminerUtility().run()
