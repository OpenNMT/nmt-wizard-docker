
class TranslationUnit(object):
    """Class to store information about translation units."""
    def __init__(self, src_line, tgt_line):
        self.src_raw = src_line
        self.tgt_raw = tgt_line

        # TODO : do we need to keep raw separately or should it be same ?
        self.__src_detok = self.src_raw
        self.__tgt_detok = self.tgt_raw

        # TODO is this really necessary ?
        self.__src_tok = None
        self.__tgt_tok = None

        # TODO : is it costly to make as many references as TUs ?
        # Should it be set on batch level ?
        self.__src_tokenizer = None
        self.__tgt_tokenizer = None

    def get_src_tok(self):
        if self.__src_tok is None:
            # TODO: should tokenization always be set ?
            # raise RuntimeError('No tokenizer is set, cannot perform tokenization.')
            if self.__src_tokenizer :
                self.__src_tok,_ = self.__src_tokenizer.tokenize(self.__src_detok)
        return self.__src_tok

    def get_tgt_tok(self):
        if self.__tgt_tok is None:
            # TODO: should tokenization always be set ?
            # raise RuntimeError('No tokenizer is set, cannot perform tokenization.')
            if self.__tgt_tokenizer :
                self.__tgt_tok,_ = self.__tgt_tokenizer.tokenize(self.__tgt_detok)
        return self.__tgt_tok

    def get_src_detok(self):
        if self.__src_detok is None:
            if not self.__src_tokenizer :
                raise RuntimeError('No tokenizer is set, cannot perform detokenization.')
            self.__src_detok = self.__src_tokenizer.detokenize(self.__src_tok)
        return self.__src_detok

    def get_tgt_detok(self):
        if self.__tgt_detok is None:
            if not self.__tgt_tokenizer :
                raise RuntimeError('No tokenizer is set, cannot perform detokenization.')
            self.__tgt_detok = self.__tgt_tokenizer.detokenize(self.__tgt_tok)
        return self.__tgt_detok


    def reset_src_tok(self, tokenizer):
        if self.__src_tok :
            if not self.__src_tokenizer :
                raise RuntimeError('No tokenizer is set, cannot perform detokenization.')
            self.__src_detok = self.__src_tokenizer.detokenize(self.__src_tok)
        self.__src_tok = None
        self.__src_tokenizer = tokenizer

    def reset_tgt_tok(self, tokenizer):
        if self.__tgt_tok :
            if not self.__tgt_tokenizer :
                raise RuntimeError('No tokenizer is set, cannot perform detokenization.')
            self.__tgt_detok = self.__tgt_tokenizer.detokenize(self.__tgt_tok)
        self.__tgt_tok = None
        self.__tgt_tokenizer = tokenizer
