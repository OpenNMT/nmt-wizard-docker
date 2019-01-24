import os
import re
import tempfile
import subprocess
import json

from nmtwizard.utility import Utility
from nmtwizard.logger import get_logger

LOGGER = get_logger(__name__)

class ScoreUtility(Utility):
    def __init__(self):
        super(ScoreUtility, self).__init__()
        self._tools_dir = os.getenv('TOOLS_DIR', '/root/tools')
        self.metric_lang = {
            # comma separated if there are multi languages
            "BLEU": "all",
            "TER": "all"
            }

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
        parser_score.add_argument('-f', '--file', default='-',
                                  help='file to save score result to.')
        parser_score.add_argument('-l', '--lang', default='en',
                                  help='Lang ID')
        parser_score.add_argument('-tok', '--tok_config',
                                  help='Configuration for tokenizer')

    def convert_to_local_file(self, nextval):
        new_val = []
        for val in nextval:
            inputs = val.split(',')
            local_inputs = []
            for remote_input in inputs:
                local_input = os.path.join(self._data_dir, self._storage.split(remote_input)[-1])
                self._storage.get_file(remote_input, local_input)
                local_inputs.append(local_input)
            new_val.append(','.join(local_inputs))
        return new_val

    def check_supported_metric(self, lang):
        metric_supported = []
        for metric, lang_defined in self.metric_lang.items():
            lang_group = lang_defined.split(',')
            if 'all' in lang_group or lang in lang_group:
                metric_supported.append(metric)
        return metric_supported

    def eval_BLEU(self, tgtfile, reffile):
        reffile = reffile.replace(',', ' ')
        result = subprocess.check_output('perl %s %s < %s' % (
                                            os.path.join(self._tools_dir, 'BLEU', 'multi-bleu-detok_cjk.perl'),
                                            reffile,
                                            tgtfile), shell=True)  # nosec
        bleu = re.match(r"^BLEU\s=\s([\d\.]+),\s([\d\.]+)/", result.decode('ascii'))
        bleu_score = {}
        bleu_score['BLEU'] = float(bleu.group(1))
        bleu_score['BLEU-1'] = float(bleu.group(2))
        return bleu_score

    def eval_TER(self, tgtfile, reffile):
        reffile_group = reffile.split(',')
        with tempfile.NamedTemporaryFile(mode='w') as file_tgt, tempfile.NamedTemporaryFile(mode='w') as file_ref:
            with open(tgtfile) as f:
                for i, line in enumerate(f):
                    file_tgt.write('%s\t(%d-)\n' % (line.rstrip(), i))
            for ref in reffile_group:
                with open(ref) as f:
                    for i, line in enumerate(f):
                        file_ref.write('%s\t(%d-)\n' % (line.rstrip(), i))
            file_tgt.flush()
            file_ref.flush()
            subprocess.check_output(['perl', os.path.join(self._tools_dir, 'TER', 'tercom_v6b.pl'),
                                  '-r', file_ref.name,
                                  '-h', file_tgt.name,
                                  '-s', '-N'])
            result = subprocess.check_output(['tail', '-1', file_tgt.name+'.sys.sum'])
            ter = re.match(r"^.*?([\d\.]+)$", result.decode('ascii').rstrip())
            return float(ter.group(1))

    def exec_function(self, args):
        list_output = self.convert_to_local_file(args.output)
        list_ref = self.convert_to_local_file(args.ref)

        if len(list_output) != len(list_ref):
            raise ValueError("`--output` and `--ref` should have same number of parameters")

        metric_supported = self.check_supported_metric(args.lang)
        score = {}
        for i, output in enumerate(list_output):
            score[args.output[i]] = {}
            for metric in metric_supported:
                if metric == 'BLEU':
                    bleu_score = self.eval_BLEU(output, list_ref[i])
                    for k, v in bleu_score.items():
                        score[args.output[i]][k] = v
                if metric == 'TER':
                    score[args.output[i]][metric] = self.eval_TER(output, list_ref[i])

        # dump score to stdout, or transfer to storage as specified
        print(json.dumps(score))
        if args.file != '-':
            with tempfile.NamedTemporaryFile(mode='w') as file_handler:
                file_handler.write(json.dumps(score))
                file_handler.flush()
                self._storage.push(file_handler.name, args.file)


if __name__ == '__main__':
    ScoreUtility().run()
