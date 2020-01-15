class TranslationSide(object):

    def __init__(self):
        self.raw = None
        self.tok = []
        self.detok = None
        self.tokenizer = None

    def get_tok(self):
        if not self.tok:
            if self.tokenizer and self.detok:
                tok,_ = self.tokenizer.tokenize(self.detok)
                self.tok.append(tok)
            # TODO: ignore empty lines and reactivate this
            # else:
            #     raise RuntimeError('Cannot perform tokenization.')
        return list(self.tok)

    def set_tok(self, tokenizer, tok):
        if tok:
            # Set a new list of tokens and a new tokenizer.
            self.tok = tok
            self.tokenizer = tokenizer
            self.detok = None
        else:
            # Set a new tokenizer, perform detokenization with previous one.
            if self.tok :
                if not self.tokenizer :
                    raise RuntimeError('No tokenizer is set, cannot perform detokenization.')
                self.detok = self.tokenizer.detokenize(self.tok[0]) # TODO : preperly deal with multipart.
            self.tok = []
            self.tokenizer = tokenizer


    def get_detok(self):
        if self.detok is None:
            if self.tokenizer and self.tok:
                self.detok = self.tokenizer.detokenize(self.tok[0])
            else:
                raise RuntimeError('Cannot perform detokenization.')
        return self.detok


    def set_detok(self, detok):
        self.detok = detok
        self.tok = []

class TranslationUnit(object):
    """Class to store information about translation units."""
    def __init__(self, input, start_state=None):

        self.__source = TranslationSide()
        self.__target = None
        self.__metadata = [1] #TODO: proper metadata

        if isinstance(input, tuple):
            # We have both source and target.
            # Can be raw (in training) or tokenized and in parts (in postprocess).
            source, target = input
            self.__target = TranslationSide()
            if isinstance(source, tuple):
                source, self.__metadata = source
            if isinstance(source, list) and isinstance(target, list):
                # Postprocess.
                self.__source.tok = source
                self.__target.tok = target
                self.__source.tokenizer = start_state["src_tokenizer"]
                self.__target.tokenizer = start_state["tgt_tokenizer"]
            else:
                # Preprocess in training.
                self.__source.raw = source.strip()
                self.__target.raw = target.strip()
                self.__source.detok = self.__source.raw
                self.__target.detok = self.__target.raw
        else:
            # We have source only: preprocess at inferences.
            self.__source.raw = input.strip()
            self.__source.detok = self.__source.raw

    def get_src_tok(self):
        return self.__source.get_tok()

    def get_tgt_tok(self):
        if self.__target:
            return self.__target.get_tok()
        return self.__target


    def set_src_tok(self, tokenizer, tok=None):
        self.__source.set_tok(tokenizer, tok)

    def set_tgt_tok(self, tokenizer, tok=None):
        if self.__target:
            self.__target.set_tok(tokenizer, tok)


    def get_src_detok(self):
        return self.__source.get_detok()

    def get_tgt_detok(self):
        if self.__target:
            return self.__target.get_detok()
        return self.__target


    def set_src_detok(self, detok):
        self.__source.set_detok(detok)

    def set_tgt_detok(self, detok):
        if self.__target:
            self.__target.set_detok(detok)


    def get_meta(self):
        return self.__metadata
