import collections
import itertools
import pyonmttok

from nmtwizard.logger import get_logger
from nmtwizard.preprocess import prepoperator

logger = get_logger(__name__)


class Tokenization(object):
    """Structure to keep tokenizer and tokens together."""

    def __init__(self, tokenizer, tokens=None, token_objects=None):
        self._tokenizer = tokenizer
        self._token_objects = token_objects
        self._tokens = tokens

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def token_objects(self):
        if self._tokens is None:
            return None
        if self._token_objects is None:
            self._token_objects = [
                self._tokenizer.deserialize_tokens(part) for part in self._tokens]
        return self._token_objects

    @property
    def tokens(self):
        return self._tokens


class PreprocessOutput(collections.namedtuple("TranslationUnitExport", (
        "src",
        "tgt",
        "metadata",
        "alignment",
))):
    """Structure containing the preprocessing result."""


class TokReplace(collections.namedtuple("TokReplace", ("start_tok_idx", "tok_num", "new_tokens"))):
        """Tuple structure for replacement in tokenization."""

class Alignment(object):

    def __init__(self, aligner=None, alignments=None):
        if aligner is None and alignments is None:
            raise RuntimeError('Cannot set an empty alignment.')
        # A list of alignments, one for each part
        self.__alignments = alignments
        self.__log_probs = None
        self.aligner = aligner

    @property
    def alignments(self):
        return self.__alignments

    @property
    def log_probs(self):
        return self.__log_probs

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
            log_probs = []
            for src_tok_part, tgt_tok_part in zip(src_tok, tgt_tok):
                align_result = self.aligner.align(src_tok_part, tgt_tok_part)
                alignments.append(set(align_result["alignments"]))
                log_probs.append((
                    align_result["forward_log_prob"], align_result["backward_log_prob"]))
            self.__alignments = alignments
            self.__log_probs = log_probs

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

    def __init__(self, line, output_side, output_delimiter=None, tokenizer=None):
        self._output_side = output_side
        self._output_delimiter = output_delimiter
        if isinstance(line, bytes):
            line = line.decode("utf-8")  # Ensure Unicode string.
        if isinstance(line, str):
            self.__detok = line.strip()
            self.__tok = None
            self.__tokenizer = tokenizer
        elif isinstance(line, list):
            self.__detok = None
            # Merge multi-parts.
            tokens = list(itertools.chain.from_iterable(line))
            self.tok = (tokenizer, [tokens])
        else:
            raise TypeError("Can't build a TranslationSide from type %s" % type(line))

    @property
    def tok(self):
        if self.__tok is None:
            if self.__tokenizer is None or self.__detok is None:
                return Tokenization(self.__tokenizer)
            else:
                self.__tok = [self.__tokenizer.tokenize(self.__detok)[0]]
        return Tokenization(self.__tokenizer, list(self.__tok))

    @tok.setter
    def tok(self, tok):
        tokenizer, tok = tok
        if tok is not None:
            # Set a new list of tokens and a new tokenizer.
            if tok and tok[0] and isinstance(tok[0][0], pyonmttok.Token):
                self.__tok = [tokenizer.serialize_tokens(part)[0] for part in tok]
            else:
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

    @property
    def output_side(self):
        return self._output_side

    @property
    def output_delimiter(self):
        return self._output_delimiter

    def finalize(self):
        _ = self.tok
        _ = self.detok
        self.__tokenizer = None

    def append(self, other):
        other_token_objects = other.tok.token_objects
        if other_token_objects is None:
            if not other.detok:
                return False
            self.detok = (
                self.detok
                + ((' ' + other.output_delimiter) if other.output_delimiter is not None else '')
                + ' ' + other.detok)
        else:
            if not other_token_objects[0]:
                return False
            tok = self.tok
            tokenizer = tok.tokenizer
            token_objects = tok.token_objects
            if token_objects is None:
                token_objects = [[]]
            elif len(token_objects) > 1:
                return False
            elif token_objects[0] and other.output_delimiter is not None:
                token_objects[0].append(pyonmttok.Token(other.output_delimiter))
            token_objects[0].extend(other_token_objects[0])
            self.tok = (tokenizer, token_objects)
        return True

    def replace_tokens(self, start_idx, tok_num, new_tokens=None, part=0):

        # check/initialize tokenization if not done already
        cur_tokens = self.__tok
        if cur_tokens is not None:
            cur_tokens = self.__tokenizer.deserialize_tokens(cur_tokens[part])
            cur_length = len(cur_tokens)
            if start_idx > cur_length:
                raise IndexError('Start index is too big for replacement.')

            end_idx = start_idx + tok_num
            if end_idx > cur_length:
                raise IndexError('Too many tokens to delete.')

            if not new_tokens:  # Deletion.
                if start_idx < cur_length:
                    del cur_tokens[start_idx:end_idx]
            else:
                if not isinstance(new_tokens[0], pyonmttok.Token):
                    new_tokens = self.__tokenizer.deserialize_tokens(new_tokens)

                if start_idx == end_idx:  # Insertion.
                    cur_tokens[start_idx:start_idx] = new_tokens
                else:  # Replacement.
                    new_tokens[0].join_left = cur_tokens[start_idx].join_left
                    new_tokens[-1].join_right = cur_tokens[end_idx - 1].join_right
                    cur_tokens[start_idx:end_idx] = new_tokens

            self.__detok = None
            self.__tok[part] = self.__tokenizer.serialize_tokens(cur_tokens)[0]
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
        self.__source = {"main": TranslationSide(source, "source", tokenizer=source_tokenizer)}
        self.__target = None
        self.__metadata = metadata if metadata is not None else [None] #TODO: proper metadata
        self.__annotations = annotations
        self.__alignment = None

        if target is not None:
            self.__target = {"main": TranslationSide(target, "target", tokenizer=target_tokenizer)}
            if alignment is not None:
                self.__alignment = Alignment(alignments=alignment)

    def add_source(self, source, output_side, name, tokenizer=None, output_delimiter=None):
        ts = TranslationSide(source, output_side, tokenizer=tokenizer, output_delimiter=output_delimiter)
        if self.__source is not None:
            if name in self.__source:
                raise RuntimeError("The source named '{}' already exists.".format(name))
            self.__source[name] = ts
        else:
            self.__source = {name:ts}

    def add_target(self, target, output_side, name, tokenizer=None, output_delimiter=None):
        ts = TranslationSide(target, output_side, tokenizer=tokenizer, output_delimiter=output_delimiter)
        if self.__target is not None:
            if name in self.__target:
                raise RuntimeError("The target named '{}' already exists.".format(name))
            self.__target[name] = ts
        else:
            self.__target = {name: ts}

    @property
    def num_sources(self):
        return len(self.__source)

    @property
    def num_targets(self):
        return len(self.__target) if self.__target is not None else 0


    @property
    def src_tok(self):
        return self.get_src_tok("main")

    def get_src_tok(self, key):
        source = self.__source.get(key)
        if source is None:
            return None
        return source.tok

    def src_tok_gen(self):
        if self.__source is not None:
            for i,s in self.__source.items():
                yield i, s.tok

    @property
    def tgt_tok(self):
        return self.get_tgt_tok("main")

    def get_tgt_tok(self, key):
        if self.__target is not None:
            target = self.__target.get(key)
            if target is not None:
                return target.tok
        return None

    def tgt_tok_gen(self):
        if self.__target is not None:
            for i,t in self.__target.items():
                yield i, t.tok

    @src_tok.setter
    def src_tok(self, tok):
        self.set_src_tok(tok, "main")

    def set_src_tok(self, tok, key):
        source = self.__source.get(key)
        if source is not None:
            source.tok = tok
        if key == "main":
            self._invalidate_alignment()

    @tgt_tok.setter
    def tgt_tok(self, tok):
        self.set_tgt_tok(tok, "main")

    def set_tgt_tok(self, tok, key):
        if self.__target is not None:
            target = self.__target.get(key)
            if target is not None:
                target.tok = tok
            if key == "main":
                self._invalidate_alignment()

    @property
    def alignment(self):
        if self.__alignment is None:
            return None
        self._initialize_alignment()
        return [set(part) for part in self.__alignment.alignments]

    @property
    def alignment_log_probs(self):
        if self.__alignment is None:
            return None
        self._initialize_alignment()
        log_probs = self.__alignment.log_probs
        if log_probs is None:
            return None
        return list(log_probs)

    def set_aligner(self, aligner):
        if self.src_tok.tokenizer is None or self.tgt_tok.tokenizer is None:
            raise RuntimeError('Cannot set aligner if not tokenization is set.')
        if self.__alignment is not None:
            self.__alignment.aligner = aligner
            self.__alignment.alignments = None
        else:
            self.__alignment = Alignment(aligner)

    def _initialize_alignment(self):
        if self.__alignment is not None and self.__alignment.alignments is None:
            self.__alignment.align(self.src_tok.tokens, self.tgt_tok.tokens)

    def _invalidate_alignment(self):
        if self.__alignment is not None:
            self.__alignment.alignments = None

    @property
    def src_detok(self):
        return self.get_src_detok("main")

    def get_src_detok(self, key):
        source = self.__source.get(key)
        if source is None:
            return None
        return source.detok

    def src_detok_gen(self):
        if self.__source is not None:
            for i,s in self.__source.items():
                yield i, s.detok

    @property
    def tgt_detok(self):
        return self.get_tgt_detok("main")

    def get_tgt_detok(self, key):
        if self.__target is not None:
            target = self.__target.get(key)
            if target is not None:
                return target.detok
        return None

    def tgt_detok_gen(self):
        if self.__target is not None:
            for i,t in self.__target.items():
                yield i, t.detok

    @src_detok.setter
    def src_detok(self, detok):
        self.set_src_detok(detok, "main")

    def set_src_detok(self, detok, key):
        source = self.__source.get(key)
        if source is not None:
            source.detok = detok
        if key == "main":
            self._invalidate_alignment()


    @tgt_detok.setter
    def tgt_detok(self, detok):
        self.set_tgt_detok(detok, "main")

    def set_tgt_detok(self, detok, key):
        if self.__target is not None:
            target = self.__target.get(key)
            if target is not None:
                target.detok = detok
            if key == "main":
                self._invalidate_alignment()

    def _finalize_side(self, name, side):
        main_side = side.get("main")
        if main_side is None:
            return

        # Merge secondary sides into the main side.
        all_sides = self.__source.items()
        if self.__target is not None:
            all_sides = itertools.chain(all_sides, self.__target.items())
        sides_to_merge = (ts for k, ts in all_sides if k != "main" and ts.output_side == name)
        for ts in sides_to_merge:
            main_side.append(ts)

        # Synchronize the main side and detach the Tokenizer instance.
        main_side.finalize()

        # Remove secondary sides.
        for key in list(side.keys()):
            if key != "main":
                side.pop(key)

    def finalize(self, process_type):
        if process_type != prepoperator.ProcessType.POSTPROCESS:
            # Synchronize alignments and detach the Aligner instance.
            self._initialize_alignment()
            if self.__alignment is not None:
                self.__alignment.aligner = None

            self._finalize_side("source", self.__source)
            if self.__target is not None:
                self._finalize_side("target", self.__target)

    def export(self, process_type):
        if process_type == prepoperator.ProcessType.POSTPROCESS:
            return self.tgt_detok
        src = self.src_tok.tokens
        if src is None:
            src = self.src_detok

        tgt = None
        tgt_tok = self.tgt_tok
        if tgt_tok is not None:
            tgt = self.tgt_tok.tokens
            if tgt is None:
                tgt = self.tgt_detok

        return PreprocessOutput(
            src=src,
            tgt=tgt,
            metadata=self.metadata,
            alignment=self.alignment,
        )

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
        self._initialize_alignment()

        if side == "source":
            self.__source["main"].replace_tokens(*replacement, part=part)
            if self.__alignment is not None:
                self.__alignment.adjust_alignment(0, *replacement, part=part)
        elif side == "target":
            self.__target["main"].replace_tokens(*replacement, part=part)
            if self.__alignment is not None:
                self.__alignment.adjust_alignment(1, *replacement, part=part)
