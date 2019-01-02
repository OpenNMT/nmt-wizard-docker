import os
import six
import re
import time
import random

from nmtwizard.utility import Utility
from nmtwizard.logger import get_logger
from nmtwizard.utility import resolve_environment_variables

from similarity import main

logger = get_logger(__name__)

class SimilarityUtility(Utility):

    def __init__(self):
        super(SimilarityUtility, self).__init__()

    @property
    def name(self):
        return "similarity"

    def exec_function(self, args):
        assert len(args) and (args[0] == "train" or args[0] == "apply"), "invalid parameters for similarity module"
        if args[0] == "train":
            assert "-trn" in args, "missing `-trn` parameters in training mode"
        else:
            assert "-tst" in args and "-mdir" in args, "missing `-tst` or `-mdir` parameters in inference mode"
            assert "-output" in args, "missing `-output` parameter in inference mode"
        main(['similarity.py'] + resolve_environment_variables(args[1:]))


if __name__ == '__main__':
    SimilarityUtility().run()
