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
        parser_score.add_argument('-o', '--output', required=True, nargs='+',
                                  help='Output file from translation.')
        parser_score.add_argument('-r', '--ref', required=True, nargs='+',
                                  help='Reference file.')
        parser_score.add_argument('-l', '--lang', default='en',
                                  help='Lang ID')

    def convert_to_local_file(self, nextval):
        new_val = []
        for val in nextval:
            inputs = val.split(',')
            local_inputs = []
            for remote_input in inputs:
                local_input = os.path.join(self._data_dir, self._storage.split(remote_input)[-1])
                print("--", remote_input, local_input)
                self._storage.get_file(remote_input, local_input)
                local_inputs.append(local_input)
            new_val.append(','.join(local_inputs))
        return new_val

    def eval_BLEU(self, tgtfile, reffile):
        reffile = reffile.replace(',', ' ')
        result = subprocess.check_output('perl ' + self._tools_dir +
                                         '/BLEU/multi-bleu-detok_cjk.perl ' + reffile +
                                         ' < ' + tgtfile, shell=True)  # nosec
        bleu = re.match("^BLEU\s=\s([\d\.]+),", result.decode('ascii'))
        return bleu.group(1)

    def exec_function(self, args):
        val_tgt = self.convert_to_local_file(args.output)
        val_ref = self.convert_to_local_file(args.ref)

        score = {}
        for i in range(len(val_tgt)):
            tgt_base = re.match("^.*/([^/]*)$", val_tgt[i]).group(1)
            score[tgt_base] = {}
            score[tgt_base]['BLEU'] = self.eval_BLEU(val_tgt[i], val_ref[i])

        print(json.dumps(score))


if __name__ == '__main__':
    ScoreUtility().run()
