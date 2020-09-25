from nmtwizard.preprocess import prepoperator

@prepoperator.register_operator("length_filter")
class LengthFilter(prepoperator.Filter):

    def __init__(self, config, process_type, build_state):

        super(LengthFilter, self).__init__()

        self._source_max = config.get('source', {}).get('max_length_char')
        self._target_max = config.get('target', {}).get('max_length_char')

        if self._source_max:
            self._criteria.append(lambda x:len(x.src_detok) > self._source_max)

        if self._target_max:
            self._criteria.append(lambda x:len(x.tgt_detok) > self._target_max)
