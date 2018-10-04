import os
import six
import re
import requests

from nmtwizard.framework import Framework
from nmtwizard.logger import get_logger
from nmtwizard.serving import pick_free_port, TranslationOutput
from nmtwizard.utils import run_cmd

logger = get_logger(__name__)


class OpenNMTLuaFramework(Framework):

    def __init__(self):
        super(OpenNMTLuaFramework, self).__init__(support_multi_training_files=True)
        self._onmt_dir = os.getenv('ONMT_DIR', '/root/opennmt')

    def train_multi_files(self,
                          config,
                          data_dir,
                          src_vocab_info,
                          tgt_vocab_info,
                          model_path=None,
                          num_samples=None,
                          samples_metadata=None,
                          gpuid=0):
        if isinstance(gpuid, list):
            logger.warning('no support of multi-gpu for opennmt_lua train')
            gpuid = gpuid[0]
        options = self._get_training_options(
            config,
            data_dir,
            src_vocab_info,
            tgt_vocab_info,
            model_path=model_path,
            num_samples=num_samples,
            samples_metadata=samples_metadata,
            gpuid=gpuid)
        options = _build_cmd_line_options(options)

        self._run_command(["th", "train.lua"] + options)

        # find output epochs - should be only one
        outputs = os.listdir(self._output_dir)
        models = filter(re.compile('.*epoch.*.t7$').search, outputs)
        # there should be only one
        if not models:
            raise RuntimeError('no model generated by the training')
        if len(models) > 1:
            raise RuntimeError('more than one model generated by the training')

        model_file = os.path.join(self._output_dir, models[0])
        return {"model.t7": model_file}

    def train(self, *arg, **kwargs):
        raise NotImplementedError()

    def trans(self, config, model_path, input, output, gpuid=0):
        if isinstance(gpuid, list):
            logger.warning('no support of multi-gpu for opennmt_lua trans')
            gpuid = gpuid[0]
        model_file = os.path.join(model_path, 'model_released.t7')
        options = self._get_translation_options(
            config, model_file, input=input, output=output, gpuid=gpuid)
        options = _build_cmd_line_options(options)
        self._run_command(["th", "translate.lua"] + options)

    def release(self, config, model_path, gpuid=0):
        model_file = os.path.join(model_path, "model.t7")
        released_model_file = os.path.join(model_path, "model_released.t7")
        release_options = self._get_release_options(model_file, released_model_file, gpuid=gpuid)
        release_options = _build_cmd_line_options(release_options)
        self._run_command(["th", "tools/release_model.lua"] + release_options)
        if not os.path.isfile(released_model_file):
            raise RuntimeError("failed to release the training model")
        return {"model_released.t7": released_model_file}

    def serve(self, config, model_path, gpuid=0):
        if isinstance(gpuid, list):
            logger.warning('no support of multi-gpu for opennmt_lua serve')
            gpuid = gpuid[0]
        model_file = os.path.join(model_path, 'model_released.t7')
        host_ip = '127.0.0.1'
        port = pick_free_port()
        options = self._get_translation_options(config, model_file, gpuid=gpuid)
        options['host'] = host_ip
        options['port'] = port
        options['withAttn'] = 'true'
        options['mode'] = 'space'
        options = _build_cmd_line_options(options)
        process = self._run_command(
            ['th', 'tools/rest_translation_server.lua'] + options, background=True)
        info = {'endpoint': 'http://{}:{}/translator/translate'.format(host_ip, port)}
        return process, info

    def forward_request(self, batch_inputs, info, timeout=None):
        batch_inputs = [{'src': ' '.join(tokens)} for tokens in batch_inputs]
        try:
            batch_results = requests.post(
                info['endpoint'], json=batch_inputs, timeout=timeout).json()
        except requests.exceptions.Timeout as e:
            logger.error('%s', e)
            return None
        batch_outputs = []
        for hypotheses in batch_results:
            outputs = []
            for hyp in hypotheses:
                tokens = hyp['tgt'].split()
                score = hyp['pred_score'] / len(tokens)
                outputs.append(TranslationOutput(tokens, score=score, attention=hyp['attn']))
            batch_outputs.append(outputs)
        return batch_outputs

    def _get_training_options(self,
                              config,
                              data_dir,
                              src_vocab_info,
                              tgt_vocab_info,
                              model_path=None,
                              num_samples=None,
                              samples_metadata=None,
                              gpuid=0):
        options = {}
        options.update(config["options"].get("common", {}))
        options.update(config["options"].get("train", {}))
        options['train_dir'] = data_dir
        options['src_vocab'] = src_vocab_info["current"]
        options['tgt_vocab'] = tgt_vocab_info["current"]
        options['update_vocab'] = (
            'replace' if src_vocab_info['changed'] or tgt_vocab_info['changed'] else 'none')
        options['report_every'] = '1000'
        options['src_suffix'] = config['source']
        options['tgt_suffix'] = config['target']
        options['gpuid'] = gpuid
        options['save_model'] = "%s/trained_model" % self._output_dir
        options['start_epoch'] = options.get('start_epoch', '1')
        options['end_epoch'] = options.get('end_epoch', '1')
        if model_path is not None:
            options['train_from'] = os.path.join(model_path, 'model.t7')
        if num_samples is not None:
            rule_file = os.path.join(self._onmt_dir, 'rule')
            _generate_distribution_file(rule_file, samples_metadata)
            options['gsample_dist'] = rule_file
            options['gsample'] = num_samples
        return options

    def _get_translation_options(self, config, model_file, input=None, output=None, gpuid=0):
        options = {}
        options.update(config["options"].get("common", {}))
        options.update(config["options"].get("trans", {}))
        options['gpuid'] = gpuid
        options['model'] = model_file
        if output is not None:
            options['output'] = output
        if input is not None:
            options['src'] = input
        return options

    def _get_release_options(self, model_file, released_model_file, gpuid=0):
        options = {}
        options["model"] = model_file
        options["output_model"] = released_model_file
        options["gpuid"] = gpuid
        return options

    def _map_vocab_entry(self, index, token, vocab):
        if index == 0:
            vocab.write(b"<blank> 1\n")
            vocab.write(b"<unk> 2\n")
            vocab.write(b"<s> 3\n")
            vocab.write(b"</s> 4\n")
        vocab.write(b"%s %d\n" % (line.strip(), index + 5))

    def _run_command(self, cmd, background=False):
        return run_cmd(cmd, cwd=self._onmt_dir, background=background)


def _protect_characters(string, protected, protector='%'):
    new_string = ""
    for char in string:
        if protected.find(char) != -1:
            new_string += protector + char
        else:
            new_string += char
    return new_string

def _generate_distribution_file(path, metadata):
    logger.debug('Writing distribution rule file to %s', path)
    with open(path, 'w') as rule_file:
        for pattern, arg in six.iteritems(metadata):
            rule_file.write('%s *' % _protect_characters(pattern, '.*+-?%', protector='%'))
            if arg is not None:
                rule_file.write(' %s' % arg)
            rule_file.write('\n')

def _build_cmd_line_options(options):
    opts = []
    for key, value in six.iteritems(options):
        if key is not None:
            opts.append('-%s' % key)
            opts.append(str(value))
    return opts


if __name__ == "__main__":
    OpenNMTLuaFramework().run()
