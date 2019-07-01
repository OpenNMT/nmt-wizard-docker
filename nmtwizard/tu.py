
class TranslationUnit(object):
    """Class to store information about translation units."""
    def __init__(self, src_line, tgt_line):
        self.src_raw = src_line
        self.tgt_raw = tgt_line
