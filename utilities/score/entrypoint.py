import os
import re
import tempfile
import subprocess
import json
import time

from nmtwizard.utility import Utility
from nmtwizard.logger import get_logger
from nmtwizard import tokenizer

LOGGER = get_logger(__name__)

class ScoreUtility(Utility):
    def __init__(self):
        super(ScoreUtility, self).__init__()
        self._tools_dir = os.getenv('TOOLS_DIR', '/root/tools')
        self.metric_lang = {
            # comma separated if there are multi languages
            "BLEU": "all",
            "TER": "all",
            "Otem-Utem": "all",
            "NIST": "all"
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

    def check_supported_metric(self, lang):
        metric_supported = []
        for metric, lang_defined in self.metric_lang.items():
            lang_group = lang_defined.split(',')
            if 'all' in lang_group or lang in lang_group:
                metric_supported.append(metric)
        return metric_supported

    def build_tokenizer_by_config(self, tok_config, lang):
        if tok_config is None:
            tok_config = {"mode": "aggressive"}
            if lang == 'zh':
              tok_config['segment_alphabet'] = ['Han']
              tok_config['segment_alphabet_change'] = True
        # to avoid SentencePiece sampling
        if 'sp_nbest_size' in tok_config:
            tok_config['sp_nbest_size'] = 0
        return tokenizer.build_tokenizer(tok_config)

    def tokenize_files(self, file_list, lang_tokenizer):
        outfile_group = []
        for file in file_list:
            outfile = tempfile.NamedTemporaryFile(delete=False)
            tokenizer.tokenize_file(lang_tokenizer, file, outfile.name)
            outfile_group.append(outfile.name)
        return outfile_group

    def exec_command_with_timeout(self, cmd, timeout=300, shell=False):
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=shell)

        iterations = 0
        while p.poll() is None and iterations < timeout:
            iterations += 1
            time.sleep(1)
        if p.poll() is None:
            print("Time out, kill process...")
            p.kill()

        return p.stdout.read()

    def eval_BLEU(self, tgtfile, reffile):
        reffile = reffile.replace(',', ' ')
        result = self.exec_command_with_timeout('/usr/bin/perl %s %s < %s' % (
                                            os.path.join(self._tools_dir, 'BLEU', 'multi-bleu-detok_cjk.perl'),
                                            reffile,
                                            tgtfile), shell=True)  # nosec
        bleu = re.match(r"^BLEU\s=\s([\d\.]+),\s([\d\.]+)/", result.decode('ascii'))
        bleu_score = {'BLEU': 0, 'BLEU-1': 0}
        if bleu is not None:
            bleu_score['BLEU'] = float(bleu.group(1))
            bleu_score['BLEU-1'] = float(bleu.group(2))
        return bleu_score

    def eval_TER(self, tgtfile, reffile):
        ter = None
        with tempfile.NamedTemporaryFile(mode='w') as file_tgt, tempfile.NamedTemporaryFile(mode='w') as file_ref:
            with open(tgtfile) as f:
                for i, line in enumerate(f):
                    file_tgt.write('%s\t(%d-)\n' % (line.rstrip(), i))
            for ref in reffile:
                with open(ref) as f:
                    for i, line in enumerate(f):
                        file_ref.write('%s\t(%d-)\n' % (line.rstrip(), i))
            file_tgt.flush()
            file_ref.flush()
            result = self.exec_command_with_timeout(['/usr/bin/java', '-jar',
                                  os.path.join(self._tools_dir, 'TER', 'tercom.7.25.jar'),
                                  '-r', file_ref.name,
                                  '-h', file_tgt.name,
                                  '-s', '-N'])
            ter = re.match(r"^.*Total\sTER:\s([\d\.]+).*$", result.decode('ascii'), re.DOTALL)
        if ter is not None:
            return round(float(ter.group(1))*100, 2)

        return 0

    def eval_Otem_Utem(self, tgtfile, reffile):
        reffile_prefix = reffile[0] + 'prefix'
        for idx, f in enumerate(reffile):
            subprocess.call(['/bin/ln', '-s', f, '%s%d' % (reffile_prefix, idx)])

        otem_utem_score = {'OTEM': 0, 'UTEM': 0}
        result = self.exec_command_with_timeout(['/usr/bin/python',
                                            os.path.join(self._tools_dir, 'Otem-Utem', 'multi-otem.py'),
                                            tgtfile,
                                            reffile_prefix])
        otem = re.match(r"^OTEM\s=\s([\d\.]+),", result.decode('ascii'))
        if otem is not None:
            otem_utem_score['OTEM'] = float(otem.group(1))

        result = self.exec_command_with_timeout(['/usr/bin/python',
                                            os.path.join(self._tools_dir, 'Otem-Utem', 'multi-utem.py'),
                                            tgtfile,
                                            reffile_prefix])
        utem = re.match(r"^UTEM\s=\s([\d\.]+),", result.decode('ascii'))
        if utem is not None:
            otem_utem_score['UTEM'] = float(utem.group(1))

        for idx in range(len(reffile)):
            os.remove('%s%d' % (reffile_prefix, idx))

        return otem_utem_score

    def eval_NIST_create_tempfile(self, typename, filename, inputfiles):
        file_handle = open(filename, "wb")
        subprocess.call(['/usr/bin/perl',
                                            os.path.join(self._tools_dir, 'NIST', 'xml_wrap.pl'),
                                            typename] + inputfiles,
                                            stdout=file_handle)

    def eval_NIST(self, tgtfile, reffile):
        file_prefix = tempfile.NamedTemporaryFile(delete=False)
        file_src_xml = file_prefix.name + '_src.xml'
        self.eval_NIST_create_tempfile('src', file_src_xml, [tgtfile])
        file_tst_xml = file_prefix.name + '_tst.xml'
        self.eval_NIST_create_tempfile('tst', file_tst_xml, [tgtfile])
        file_ref_xml = file_prefix.name + '_ref.xml'
        self.eval_NIST_create_tempfile('ref', file_ref_xml, reffile)

        result = self.exec_command_with_timeout(['/usr/bin/perl',
                                            os.path.join(self._tools_dir, 'NIST', 'mteval-v14.pl'),
                                            '-s', file_src_xml,
                                            '-t', file_tst_xml,
                                            '-r', file_ref_xml])
        nist = re.match(r"^.*NIST\sscore\s=\s([\d\.]+).*$", result.decode('ascii'), re.DOTALL)

        os.remove(file_prefix.name)
        os.remove(file_src_xml)
        os.remove(file_tst_xml)
        os.remove(file_ref_xml)

        if nist is not None:
            return float(nist.group(1))

        return 0

    def exec_function(self, args):
        list_output = self.convert_to_local_file(args.output)
        list_ref = self.convert_to_local_file(args.ref)

        if len(list_output) != len(list_ref):
            raise ValueError("`--output` and `--ref` should have same number of parameters")

        metric_supported = self.check_supported_metric(args.lang)
        lang_tokenizer = self.build_tokenizer_by_config(args.tok_config, args.lang)

        score = {}
        for i, output in enumerate(list_output):
            output_tok = self.tokenize_files([output], lang_tokenizer)
            reffile_tok = self.tokenize_files(list_ref[i].split(','), lang_tokenizer)

            print("Starting to evaluate ... %s" % args.output[i])
            score[args.output[i]] = {}
            for metric in metric_supported:
                if metric == 'BLEU':
                    bleu_score = self.eval_BLEU(output, list_ref[i])
                    for k, v in bleu_score.items():
                        print("%s: %.2f" % (k, v))
                        score[args.output[i]][k] = v
                if metric == 'TER':
                    v = self.eval_TER(output_tok[0], reffile_tok)
                    print("%s: %.2f" % (metric, v))
                    score[args.output[i]][metric] = v
                if metric == 'Otem-Utem':
                    otem_utem_score = self.eval_Otem_Utem(output_tok[0], reffile_tok)
                    for k, v in otem_utem_score.items():
                        print("%s: %.2f" % (k, v))
                        score[args.output[i]][k] = v
                if metric == 'NIST':
                    v = self.eval_NIST(output_tok[0], reffile_tok)
                    print("%s: %.2f" % (metric, v))
                    score[args.output[i]][metric] = v

            os.remove(output_tok[0])
            for file in reffile_tok:
                os.remove(file)

        # dump score to stdout, or transfer to storage as specified
        print(json.dumps(score))
        if args.file != '-':
            with tempfile.NamedTemporaryFile(mode='w') as file_handler:
                file_handler.write(json.dumps(score))
                file_handler.flush()
                self._storage.push(file_handler.name, args.file)


if __name__ == '__main__':
    ScoreUtility().run()
