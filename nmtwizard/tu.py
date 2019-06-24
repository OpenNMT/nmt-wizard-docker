
class TranslationUnit:
    """Class to store information about translation units."""
    def __init__(self, src_line, tgt_line, occurences = 1):
        self._src_raw = src_line
        self._tgt_raw = tgt_line
        self._occurences = occurences
