
class TranslationUnit(object):
    """Class to store information about translation units."""
    def __init__(self, input, tokenized=False, src_tokenizer=None, tgt_tokenizer=None):

        self.tgt_raw = None
        if isinstance(input, tuple):
            self.src_raw = input[0]
            if len(input) > 1 :
                self.tgt_raw = input[1]
        else:
            self.src_raw = input

        # TODO : do we need to keep raw separately or should it be same ?
        if not tokenized:
            self.__src_detok = self.src_raw
            self.__tgt_detok = self.tgt_raw
        else:
            self.__src_detok = None
            self.__tgt_detok = None

        # TODO is this really necessary ?
        self.__src_tok = None
        self.__tgt_tok = None

        if tokenized:
            if isinstance(self.src_raw, list):
                self.__src_tok = self.src_raw
            else:
                self.__src_tok = self.src_raw.split()

            if self.tgt_raw:
                if isinstance(self.tgt_raw, list):
                    self.__tgt_tok = self.tgt_raw
                else:
                    self.__tgt_tok = self.tgt_raw.split()
            
        # TODO : is it costly to make as many references as TUs ?
        # Should it be set on batch level ?
        self.__src_tokenizer = src_tokenizer
        self.__tgt_tokenizer = tgt_tokenizer

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
            # TODO: should tokenization always be set ?
            # if not self.__src_tokenizer :
            #     raise RuntimeError('No tokenizer is set, cannot perform detokenization.')
            if self.__src_tokenizer :
                self.__src_detok = self.__src_tokenizer.detokenize(self.__src_tok)
        return self.__src_detok

    def get_tgt_detok(self):
        if self.__tgt_detok is None:
            # if not self.__tgt_tokenizer :
            #     raise RuntimeError('No tokenizer is set, cannot perform detokenization.')
            if self.__tgt_tokenizer :
                self.__tgt_detok = self.__tgt_tokenizer.detokenize(self.__tgt_tok)
        return self.__tgt_detok

    # TODO : get rid of identical functions
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
