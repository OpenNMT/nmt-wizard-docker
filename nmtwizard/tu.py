
class TranslationUnit:
    """Class to store information about translation units."""
    def __init__(self, src_line, tgt_line):
        self._src_raw = src_line
        self._tgt_raw = tgt_line
