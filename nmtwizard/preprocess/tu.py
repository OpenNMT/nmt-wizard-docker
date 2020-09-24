import collections
import pyonmttok

from nmtwizard.preprocess import tokenizer

from nmtwizard.logger import get_logger

logger = get_logger(__name__)

class Tokenization(collections.namedtuple("Tokenization", ("tokenizer", "tokens", "token_objects"))):
    """Tuple structure to keep tokenizer and tokens together."""

class TokReplace(collections.namedtuple("TokReplace", ("start_tok_idx", "tok_num", "new_tokens"))):
        """Tuple structure for replacement in tokenization."""

class Alignment(object):

    def __init__(self, aligner=None, alignments=None):
        if not aligner and not alignments:
            raise RuntimeError('Cannot set an empty alignment.')
        self.__alignments = alignments
        self.__aligner = aligner

    @property
    def alignments(self):
        return self.__alignments

    @alignments.setter
    def alignments(self, alignments):
        if isinstance(alignments, str):
            # Initialize from pharaoh format
            # TODO : add checks.
            self.__alignments = {tuple(al.split('-')) for al in alignments.split()}
        elif isinstance(alignments, list):
            # Initialize from a list of tuples
            self.__alignments = {tuple(al) for al in alignments}
        else:
            self.__alignments = alignments

    def align(self, src_tok, tgt_tok):
        if self.__alignments is None and self.__aligner is not None:
            align_result = self.__aligner.align(src_tok, tgt_tok)
            # TODO : write fwd and bwd probs
            self.__alignments = align_result["alignments"]

    def adjust_alignment(self, side_idx, start_idx, tok_num, new_tokens=None):
        # Shift alignments behind insertion/deletion.
        # Remove alignments to deleted tokens.
        # Align dangling alignments to the first token.

        opp_side_idx = not side_idx
        # Check there is an alignment
        if self.__alignments is not None:
            new_alignment = set()

            for al in self.__alignments:
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
            self.__alignments = new_alignment


class TranslationSide(object):

    def __init__(self, element, tokenizer=None):
        self.__tok = None
        self.__tokenizer = None
        if isinstance(element, list):
            self.raw = None
            self.__detok = None
            self.tok = (tokenizer, element)
        else:
            self.raw = element.strip()
            self.__detok = self.raw

    @property
    def tok(self):
        if self.__tok is None:
            if self.__tokenizer is None:
                return Tokenization(tokenizer=None, tokens=None, token_objects=None)
            token_objects = self.__tokenizer.tokenize(self.__detok, as_token_objects=True)
            tokens, _ = self.__tokenizer.serialize_tokens(token_objects)
            self.__tok = (tokens, token_objects)
        else:
            tokens, token_objects = self.__tok
        return Tokenization(
            tokenizer=self.__tokenizer,
            tokens=list(tokens),
            token_objects=list(token_objects))

    @tok.setter
    def tok(self, tok):
        tokenizer, tokens = tok
        if tokens is None:
            if self.__tokenizer is None:
                self.__tok = None
            elif tokenizer is not self.__tokenizer and self.__tok is not None:
                # Tokenization has changed, perform detokenization with previous tokenizer.
                _, token_objects = self.__tok
                self.__detok = self.__tokenizer.detokenize(token_objects)
                self.__tok = None
        else:
            # Set a new list of tokens and a new tokenizer.
            if tokenizer is None:
                raise ValueError('A tokenizer should be declared when setting tokens')
            if tokens and isinstance(tokens[0], pyonmttok.Token):
                token_objects = tokens
                tokens, _ = tokenizer.serialize_tokens(token_objects)
            else:
                token_objects = tokenizer.deserialize_tokens(tokens)
            self.__detok = None
            self.__tok = (tokens, token_objects)
        self.__tokenizer = tokenizer

    @property
    def detok(self):
        if self.__detok is None:
            if self.__tokenizer is not None and self.__tok is not None:
                _, token_objects = self.__tok
                self.__detok = self.__tokenizer.detokenize(token_objects)
            else:
                raise RuntimeError('Cannot perform detokenization.')
        return self.__detok

    @detok.setter
    def detok(self, detok):
        self.__detok = detok
        self.__tok = None


    def replace_tokens(self, start_idx, tok_num, new_tokens=None):

        # check/initialize tokenization if not done already
        tokenizer, _, cur_tokens = self.tok
        if cur_tokens is not None:
            if start_idx > len(cur_tokens):
                raise IndexError('Start index is too big for replacement.')

            end_idx = start_idx + tok_num
            if end_idx > len(cur_tokens):
                raise IndexError('Too many tokens to delete.')

            # If we replace some tokens, check if they start or end with a joiner.
            if new_tokens:
                new_tokens = [
                    token if isinstance(token, pyonmttok.Token) else pyonmttok.Token(token)
                    for token in new_tokens]
                if start_idx != end_idx:
                    if start_idx < len(cur_tokens):
                        new_tokens[0].join_left = cur_tokens[start_idx].join_left
                    if end_idx <= len(cur_tokens):
                        new_tokens[-1].join_right = cur_tokens[end_idx - 1].join_right

            # Insert new tokens.
            for i, idx in enumerate(range(start_idx, start_idx + len(new_tokens))):
                if idx < end_idx :
                    # replace existing tokens
                    cur_tokens[idx] = new_tokens[i]
                else:
                    # insert remaining tokens
                    cur_tokens[idx:idx] = new_tokens[i:]
                    break

            # Remove remaining tokens if deletion is bigger than insertion
            if end_idx > start_idx + len(new_tokens):
                del cur_tokens[start_idx + len(new_tokens):end_idx]

            self.tok = (tokenizer, cur_tokens)
        else:
            logger.warning("Cannot replace tokens, no tokenization is set.")


class TranslationUnitPart(object):
    """Class to store information about translation units."""

    def __init__(self,
                 source,
                 target=None,
                 metadata=None,
                 annotations=None,
                 alignment=None,
                 source_tokenizer=None,
                 target_tokenizer=None):
        self.__metadata = metadata
        self.__annotations = annotations

        self.__source = TranslationSide(source, tokenizer=source_tokenizer)
        self.__target = None
        self.__alignment = None

        if target is not None:
            self.__target = TranslationSide(target, tokenizer=target_tokenizer)
            if alignment is not None:
                self.__alignment = Alignment(alignments=alignment)

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
        self._reset_align()

    @tgt_tok.setter
    def tgt_tok(self, tok):
        if self.__target is not None:
            self.__target.tok = tok
            self._reset_align()

    @property
    def alignment(self):
        if self.__alignment is None:
            return None
        if self.__alignment.alignments is None:
            self.__alignment.align(self.src_tok.tokens, self.tgt_tok.tokens)
        return self.__alignment.alignments.copy()

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
        self._reset_align()

    @tgt_detok.setter
    def tgt_detok(self, detok):
        if self.__target is not None:
            self.__target.detok = detok
            self._reset_align()

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
                       ):

        # Replace (delete, insert) tokens in a TU and adjust alignment without retoknization/realignment.

        if src_replace:
            # replace tokens in source and adjust alignment if any
            src_replace = TokReplace(*src_replace)
            self.replace_tokens_side("source", src_replace)

        if tgt_replace:
            # replace tokens in source and adjust_alignment if any
            tgt_replace = TokReplace(*tgt_replace)
            self.replace_tokens_side("target", tgt_replace)

        # TODO
        # Maybe provide and alignment for inserted tokens ?


    def replace_tokens_side(self, side, replacement):

        # Initialize alignment
        alignment = self.alignment

        if side == "source":
            self.__source.replace_tokens(*replacement)
            if alignment:
                self.__alignment.adjust_alignment(0, *replacement)
        elif side == "target":
            self.__target.replace_tokens(*replacement)
            if alignment:
                self.__alignment.adjust_alignment(1, *replacement)

    def _reset_align(self):
        if self.__alignment is not None:
            self.__alignment.alignments = None


class TranslationUnit(object):

    def __init__(self,
                 source,
                 target=None,
                 metadata=None,
                 start_state=None,
                 annotations=None,
                 alignment=None,
                 source_tokenizer=None,
                 target_tokenizer=None):
        if isinstance(source, list) and source and isinstance(source[0], list):
            num_parts = len(source)
            self.__parts = [
                TranslationUnitPart(
                    source[i],
                    target=target[i] if target else None,
                    metadata=metadata[i] if metadata else None,
                    annotations=annotations[i] if annotations else None,
                    alignment=alignment[i] if alignment else None,
                    source_tokenizer=source_tokenizer,
                    target_tokenizer=target_tokenizer)
                for i in range(num_parts)]
        else:
            self.__parts = [
                TranslationUnitPart(
                    source,
                    target=target,
                    metadata=metadata,
                    annotations=annotations,
                    alignment=alignment,
                    source_tokenizer=source_tokenizer,
                    target_tokenizer=target_tokenizer)]

    @property
    def parts(self):
        return self.__parts

    def is_multipart(self):
        return len(self.__parts) > 1

    def synchronize(self):
        for part in self.__parts:
            _ = part.src_tok
            _ = part.tgt_tok
            _ = part.alignment

    def set_aligner(self, aligner):
        for part in self.__parts:
            part.set_aligner(aligner)


    # The methods below are helpers for the common case of a single part TU.

    @property
    def src_tok(self):
        self._assert_single_part()
        return self.__parts[0].src_tok

    @property
    def tgt_tok(self):
        self._assert_single_part()
        return self.__parts[0].tgt_tok

    @src_tok.setter
    def src_tok(self, tok):
        self._assert_single_part()
        self.__parts[0].src_tok = tok

    @tgt_tok.setter
    def tgt_tok(self, tok):
        self._assert_single_part()
        self.__parts[0].tgt_tok = tok

    @property
    def alignment(self):
        self._assert_single_part()
        return self.__parts[0].alignment

    @property
    def src_detok(self):
        self._assert_single_part()
        return self.__parts[0].src_detok

    @property
    def tgt_detok(self):
        self._assert_single_part()
        return self.__parts[0].tgt_detok

    @src_detok.setter
    def src_detok(self, detok):
        self._assert_single_part()
        self.__parts[0].src_detok = detok

    @tgt_detok.setter
    def tgt_detok(self, detok):
        self._assert_single_part()
        self.__parts[0].tgt_detok = detok

    @property
    def metadata(self):
        self._assert_single_part()
        return self.__parts[0].metadata

    @metadata.setter
    def metadata(self, metadata):
        self._assert_single_part()
        self.__parts[0].metadata = metadata

    @property
    def annotations(self):
        self._assert_single_part()
        return self.__parts[0].annotations

    def replace_tokens(self, src_replace=None, tgt_replace=None):
        self._assert_single_part()
        return self.__parts[0].replace_tokens(src_replace=src_replace, tgt_replace=tgt_replace)

    def replace_tokens_side(self, side, replacement):
        self._assert_single_part()
        return self.__parts[0].replace_tokens_side(side, replacement)

    def _assert_single_part(self):
        if self.is_multipart():
            raise RuntimeError("This TU has multiple parts, but you used it as a single-part TU. "
                               "You probably need to update your operator to iterate over each "
                               "part of the TU.")
