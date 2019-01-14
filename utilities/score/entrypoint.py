import os
import re
import subprocess
import json

from nmtwizard.utility import Utility
from nmtwizard.logger import get_logger

logger = get_logger(__name__)

class ScoreUtility(Utility):

    def __init__(self):
        super(ScoreUtility, self).__init__()
        self._tools_dir = os.getenv('TOOLS_DIR', '/root/tools')

    @property
    def name(self):
        return "score"

    def declare_arguments(self, parser):
        subparsers = parser.add_subparsers(help='Run type', dest='cmd')
        parser_score = subparsers.add_parser('score', help='Evaluate translation.')
        parser_score.add_argument('-i', '--input', required=True,
                                  help='Input file.')
        parser_score.add_argument('-r', '--ref', required=True,
                                  help='Reference file.')
        parser_score.add_argument('-l', '--lang', default='en',
                                  help='Lang ID')

    def eval_BLEU(self, tgtfile, reffile):
        result = subprocess.check_output('perl ' + self._tools_dir + '/BLEU/multi-bleu-detok_cjk.perl ' + reffile + ' < ' + tgtfile, shell=True)
        bleu = re.match("^BLEU\s=\s([\d\.]+),", result.decode('ascii'))
        return bleu.group(1)

    def exec_function(self, args):
        val_tgt = args.input
        val_ref = args.ref

        score = {}
        score['BLEU'] = self.eval_BLEU(val_tgt, val_ref)

        print(json.dumps(score))


if __name__ == '__main__':
    ScoreUtility().run()
