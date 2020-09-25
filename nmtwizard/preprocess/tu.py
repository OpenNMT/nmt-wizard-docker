import collections
import copy

from nmtwizard.preprocess import tokenizer

from nmtwizard.logger import get_logger

logger = get_logger(__name__)

class Tokenization(collections.namedtuple("Tokenization", ("tokenizer", "tokens"))):
    """Tuple structure to keep tokenizer and tokens together."""

class TokReplace(collections.namedtuple("TokReplace", ("start_tok_idx", "tok_num", "new_tokens"))):
        """Tuple structure for replacement in tokenization."""

class Alignment(object):

    def __init__(self, aligner=None, alignments=None):
        if aligner is None and alignments is None:
            raise RuntimeError('Cannot set an empty alignment.')
        # A list of alignments, one for each part
        self.__alignments = alignments
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
                    self.__alignments[i] = {tuple(al.split('-')) for al in part.split()}
                elif isinstance(part, list):
                    # Initialize from a list of tuples
                    self.__alignments[i] = {tuple(al) for al in part}
                else:
                    break

    def align(self, src_tok, tgt_tok):
        if self.__alignments is None and self.aligner is not None:
            alignments = []
            for src_tok_part, tgt_tok_part in zip(src_tok, tgt_tok):
                align_result = self.aligner.align(src_tok_part, tgt_tok_part)
                # TODO : write fwd and bwd probs
                alignments.append(align_result["alignments"])
            self.__alignments = alignments

    def adjust_alignment(self, side_idx, start_idx, tok_num, new_tokens=None, part = 0):
        # Shift alignments behind insertion/deletion.
        # Remove alignments to deleted tokens.
        # Align dangling alignments to the first token.

        opp_side_idx = not side_idx
        # Check there is an alignment
        if self.__alignments is not None:
            new_alignment = set()

            for al in self.__alignments[part]:
                side_tok_idx = al[side_idx]
                opp_side_tok_idx = al[opp_side_idx]
                if side_tok_idx >= start_idx:
                    if side_tok_idx >= start_idx + tok_num:
                        # Shift alignment
                        side_tok_idx += len(new_tokens) - tok_num
                    else:
                        if len(new_tokens) == 0:
                            # Delete alignment
                            continue
                        else:
                            # Realign to the first inserted token
                            side_tok_idx = start_idx

                if side_idx:
                    new_alignment.add((opp_side_tok_idx, side_tok_idx))
                else:
                    new_alignment.add((side_tok_idx, opp_side_tok_idx))
            self.__alignments[part] = new_alignment


class TranslationSide(object):

    def __init__(self, line, tokenizer=None):
        if isinstance(line, bytes):
            line = line.decode("utf-8")  # Ensure Unicode string.
        if isinstance(line, str):
            self.raw = line.strip()
            self.__detok = self.raw
            self.__tok = None
        elif isinstance(line, list):
            self.raw = None
            self.__detok = None
            self.__tok = line
        else:
            raise TypeError("Can't build a TranslationSide from type %s" % type(line))
        self.__tokenizer = tokenizer

    @property
    def tok(self):
        if self.__tok is None:
            if self.__tokenizer is None or self.__detok is None:
                return Tokenization(tokenizer=self.__tokenizer, tokens=None)
            else:
                tok,_ = self.__tokenizer.tokenize(self.__detok)
                self.__tok = [tok]
        return Tokenization(tokenizer=self.__tokenizer, tokens=list(self.__tok))

    @tok.setter
    def tok(self, tok):
        tokenizer, tok = tok
        if tok is not None:
            # Set a new list of tokens and a new tokenizer.
            self.__tok = tok
            self.__tokenizer = tokenizer
            self.__detok = None
        else:
            # Set a new tokenizer, perform detokenization with previous one.
            if self.__tok is not None:
                if self.__tokenizer is None:
                    raise RuntimeError('No tokenizer is set, cannot perform detokenization.')
                self.__detok = self.__tokenizer.detokenize(self.__tok[0]) # TODO : preperly deal with multipart.
            self.__tok = None
            self.__tokenizer = tokenizer

    @property
    def detok(self):
        if self.__detok is None:
            if self.__tokenizer is not None and self.__tok is not None:
                self.__detok = self.__tokenizer.detokenize(self.__tok[0])
            else:
                raise RuntimeError('Cannot perform detokenization.')
        return self.__detok

    @detok.setter
    def detok(self, detok):
        self.__detok = detok
        self.__tok = None


    def replace_tokens(self, start_idx, tok_num, new_tokens=None, part=0):

        # check/initialize tokenization if not done already
        if self.tok.tokens is not None:
            if start_idx > len(self.__tok[part]):
                raise IndexError('Start index is too big for replacement.')

            end_idx = start_idx + tok_num
            if end_idx > len(self.__tok[part]):
                raise IndexError('Too many tokens to delete.')

            # If we replace some tokens, check if they start or end with a joiner.
            joiner_start = False
            joiner_end = False
            if start_idx != end_idx and new_tokens is not None:
                if start_idx < len(self.__tok[part]) and self.__tok[part][start_idx].startswith(tokenizer.joiner_marker):
                    joiner_start = True
                if end_idx <= len(self.__tok[part]) and self.__tok[part][end_idx-1].endswith(tokenizer.joiner_marker):
                    joiner_end = True

            # Insert new tokens.
            for i, idx in enumerate(range(start_idx, start_idx + len(new_tokens))):
                if idx < end_idx :
                    # replace existing tokens
                    self.__tok[part][idx] = new_tokens[i]
                else:
                    # insert remaining tokens
                    self.__tok[part][idx:idx] = new_tokens[i:]
                    break

            # Insert joiners if needed
            if joiner_start:
                self.__tok[part][start_idx] = tokenizer.joiner_marker + self.__tok[part][start_idx]
            if joiner_end:
                self.__tok[part][start_idx + len(new_tokens) - 1] = self.__tok[part][start_idx + len(new_tokens) - 1] + tokenizer.joiner_marker

            # Remove remaining tokens if deletion is bigger than insertion
            if end_idx > start_idx + len(new_tokens):
                del self.__tok[part][start_idx + len(new_tokens):end_idx]

        else:
            logger.warning("Cannot replace tokens, no tokenization is set.")


class TranslationUnit(object):
    """Class to store information about translation units."""
    def __init__(self,
                 source,
                 target=None,
                 metadata=None,
                 annotations=None,
                 alignment=None,
                 source_tokenizer=None,
                 target_tokenizer=None):
        self.__source = TranslationSide(source, tokenizer=source_tokenizer)
        self.__target = None
        self.__metadata = metadata if metadata is not None else [None] #TODO: proper metadata
        self.__annotations = annotations
        self.__alignment = None

        if target is not None:
            self.__target = TranslationSide(target, tokenizer=target_tokenizer)
            if alignment is not None:
                self.__alignment = Alignment(alignments=alignment)

    def synchronize(self):
        _ = self.src_tok
        _ = self.tgt_tok
        _ = self.alignment

    @property
    def src_tok(self):
        return self.__source.tok

    @property
    def tgt_tok(self):
        if self.__target is not None:
            return self.__target.tok
        return None


    @src_tok.setter
    def src_tok(self, tok):
        self.__source.tok = tok
        self.__alignment = None

    @tgt_tok.setter
    def tgt_tok(self, tok):
        if self.__target is not None:
            self.__target.tok = tok
            self.__alignment = None

    @property
    def alignment(self):
        if self.__alignment is None:
            return None
        if self.__alignment.alignments is None:
            self.__alignment.align(self.src_tok.tokens, self.tgt_tok.tokens)
        return copy.deepcopy(self.__alignment.alignments)

    def set_aligner(self, aligner):
        if self.src_tok.tokenizer is None or self.tgt_tok.tokenizer is None:
            raise RuntimeError('Cannot set aligner if not tokenization is set.')
        if self.__alignment is not None:
            self.__alignment.aligner = aligner
            self.__alignment.alignments = None
        else:
            self.__alignment = Alignment(aligner)

    @property
    def src_detok(self):
        return self.__source.detok

    @property
    def tgt_detok(self):
        if self.__target is not None:
            return self.__target.detok
        return None

    @src_detok.setter
    def src_detok(self, detok):
        self.__source.detok = detok
        self.__alignment = None

    @tgt_detok.setter
    def tgt_detok(self, detok):
        if self.__target is not None:
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

    def replace_tokens(self,
                       src_replace = None, # TokReplace structure
                       tgt_replace = None, # TokReplace structure
                       part = 0): # TODO : maybe rather send multi-part replacements ?

        # Replace (delete, insert) tokens in a TU and adjust alignment without retoknization/realignment.

        if src_replace:
            # replace tokens in source and adjust alignment if any
            src_replace = TokReplace(*src_replace)
            self.replace_tokens_side("source", src_replace, part=part)

        if tgt_replace:
            # replace tokens in source and adjust_alignment if any
            tgt_replace = TokReplace(*tgt_replace)
            self.replace_tokens_side("target", tgt_replace, part=part)

        # TODO
        # Maybe provide and alignment for inserted tokens ?


    def replace_tokens_side(self, side, replacement, part=0):

        # Initialize alignment
        alignment = self.alignment

        if side == "source":
            self.__source.replace_tokens(*replacement, part=part)
            if alignment:
                self.__alignment.adjust_alignment(0, *replacement, part=part)
        elif side == "target":
            self.__target.replace_tokens(*replacement, part=part)
            if alignment:
                self.__alignment.adjust_alignment(1, *replacement, part=part)
