from nmtwizard.preprocess import prepoperator
from nmtwizard.preprocess import tokenizer

@prepoperator.register_operator("tokenization")
class Tokenizer(prepoperator.Operator):

    def __init__(self, tok_config, process_type, build_state):
        self._src_tokenizer = tokenizer.build_tokenizer(tok_config["source"])
        self._tgt_tokenizer = tokenizer.build_tokenizer(tok_config["target"])

        if build_state:
            self._prev_src_tokenizer = build_state["src_tokenizer"]
            self._prev_tgt_tokenizer = build_state["tgt_tokenizer"]

            build_state["src_tokenizer"] = self._src_tokenizer
            build_state["tgt_tokenizer"] = self._tgt_tokenizer

        self._postprocess_only = build_state['postprocess_only']


    def _preprocess(self, tu_batch):
        tu_batch = self._set_tokenizers(tu_batch, self._src_tokenizer, self._tgt_tokenizer)
        return tu_batch


    def _postprocess(self, tu_batch):
        # Tokenization from 'postprocess' field applies current tokenization in postprocess.
        if self._postprocess_only:
            src_tokenizer = self._src_tokenizer
            tgt_tokenizer = self._tgt_tokenizer
        # Tokenization from 'preprocess' field applies previous tokenization in postprocess.
        else:
            src_tokenizer = self._prev_src_tokenizer
            tgt_tokenizer = self._prev_tgt_tokenizer
        tu_batch = self._set_tokenizers(tu_batch, src_tokenizer, tgt_tokenizer)
        return tu_batch


    def _set_tokenizers(self, tu_batch, src_tokenizer, tgt_tokenizer):
        tu_list, meta_batch = tu_batch

        # Set tokenizers for TUs.
        for tu in tu_list :
            tu.src_tok = (src_tokenizer, None)
            tu.tgt_tok = (tgt_tokenizer, None)

        return tu_list, meta_batch
