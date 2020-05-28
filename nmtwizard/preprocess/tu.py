import collections
import copy

from nmtwizard.preprocess import tokenizer

class Tokenization(collections.namedtuple("Tokenization", ("tokenizer", "tokens"))):
    """Tuple structure to keep tokenizer and tokens together."""

class Alignment(object):

    def __init__(self, aligner=None, alignments=None):
        if not aligner and not alignments:
            raise RuntimeError('Cannot set an empty alignment.')
        # A list of alignments, one for each part
        self.alignments = alignments
        self.aligner = aligner

    @property
    def alignments(self):
        return self.__alignments

    @alignments.setter
    def alignments(self, alignments):
        self.__alignments = alignments
        if isinstance(self.__alignments, list):
            for i,part in enumerate(self.__alignments):
                if isinstance(part, str):
                    # Initialize from pharaoh format
                    # TODO : add checks.
                    self.__alignments[i] = [tuple(al.split('-')) for al in part.split()]
                else:
                    break

    def align(self, src_tok, tgt_tok):
        if not self.alignments and self.aligner:
            alignments = []
            for src_tok_part, tgt_tok_part in zip(src_tok, tgt_tok):
                align_result = self.aligner.align(src_tok_part, tgt_tok_part)
                # TODO : write fwd and bwd probs
                alignments.append(align_result["alignments"])
            self.alignments = alignments

class TranslationSide(object):

    def __init__(self):
        self.raw = None
        self.__tok = []
        self.__detok = None
        self.__tokenizer = None

    @property
    def tok(self):
        if not self.__tok:
            if self.__tokenizer and self.__detok:
                tok,_ = self.__tokenizer.tokenize(self.__detok)
                self.__tok.append(tok)
            # TODO: ignore empty lines and reactivate this
            # else:
            #     raise RuntimeError('Cannot perform tokenization.')
        return Tokenization(tokenizer=self.__tokenizer, tokens=list(self.__tok))

    @tok.setter
    def tok(self, tok):
        tokenizer, tok = tok
        if tok:
            # Set a new list of tokens and a new tokenizer.
            self.__tok = tok
            self.__tokenizer = tokenizer
            self.__detok = None
        else:
            # Set a new tokenizer, perform detokenization with previous one.
            if self.__tok :
                if not self.__tokenizer :
                    raise RuntimeError('No tokenizer is set, cannot perform detokenization.')
                self.__detok = self.__tokenizer.detokenize(self.__tok[0]) # TODO : preperly deal with multipart.
            self.__tok = []
            self.__tokenizer = tokenizer

    @property
    def detok(self):
        if self.__detok is None:
            if self.__tokenizer and self.__tok:
                self.__detok = self.__tokenizer.detokenize(self.__tok[0])
            else:
                raise RuntimeError('Cannot perform detokenization.')
        return self.__detok

    @detok.setter
    def detok(self, detok):
        self.__detok = detok
        self.__tok = []

class TranslationUnit(object):
    """Class to store information about translation units."""
    def __init__(self, tu_input, start_state=None, annotations=None, alignment=None):

        self.__source = TranslationSide()
        self.__target = None
        self.__metadata = [None] #TODO: proper metadata
        self.__annotations = annotations
        self.__alignment = None

        if isinstance(tu_input, tuple):
            # We have both source and target.
            # Can be raw (in training, in inference with incomplete target) or tokenized and in parts (in postprocess).
            source, target = tu_input
            self.__target = TranslationSide()
            if isinstance(source, tuple):
                source, self.__metadata = source
            if isinstance(source, list) and isinstance(target, list):
                # Postprocess.
                src_tokenizer = None
                tgt_tokenizer = None
                if start_state and start_state["src_tok_config"]:
                    src_tokenizer = tokenizer.build_tokenizer(start_state["src_tok_config"])
                if start_state and start_state["tgt_tok_config"]:
                    tgt_tokenizer = tokenizer.build_tokenizer(start_state["tgt_tok_config"])
                self.__source.tok = (src_tokenizer, source)
                self.__target.tok = (tgt_tokenizer, target)
                if alignment:
                    self.__alignment = Alignment(alignments=alignment)
            else:
                # Preprocess in training or in inference with incomplete target.
                self.__source.raw = source.strip()
                self.__target.raw = target.strip()
                self.__source.detok = self.__source.raw
                self.__target.detok = self.__target.raw
        else:
            # We have source only: preprocess in inference with source only.
            self.__source.raw = tu_input.strip()
            self.__source.detok = self.__source.raw

    @property
    def src_tok(self):
        return self.__source.tok

    @property
    def tgt_tok(self):
        if self.__target:
            return self.__target.tok
        return self.__target


    @src_tok.setter
    def src_tok(self, tok):
        self.__source.tok = tok
        self.__alignment = None

    @tgt_tok.setter
    def tgt_tok(self, tok):
        if self.__target:
            self.__target.tok = tok
            self.__alignment = None

    @property
    def alignment(self):
        if not self.__alignment:
            return None
        if not self.__alignment.alignments:
            self.__alignment.align(self.src_tok.tokens, self.tgt_tok.tokens)
        return copy.deepcopy(self.__alignment.alignments)

    def set_aligner(self, aligner):
        if not self.src_tok.tokenizer or not self.tgt_tok.tokenizer:
            raise RuntimeError('Cannot set aligner if not tokenization is set.')
        if self.__alignment:
            self.__alignment.aligner = aligner
        else:
            self.__alignment = Alignment(aligner)
        self.__alignment.alignments = None

    @property
    def src_detok(self):
        return self.__source.detok

    @property
    def tgt_detok(self):
        if self.__target:
            return self.__target.detok
        return self.__target

    @src_detok.setter
    def src_detok(self, detok):
        self.__source.detok = detok
        self.__alignment = None

    @tgt_detok.setter
    def tgt_detok(self, detok):
        if self.__target:
            self.__target.detok = detok
            self.__alignment = None

    @property
    def metadata(self):
        return self.__metadata

    @metadata.setter
    def metadata(self, metadata):
        self.__metadata = metadata

    @property
    def annotations(self):
        return self.__annotations
