import os
import subprocess
import six
import shlex
import shutil

from nmtwizard.framework import Framework
from nmtwizard.logger import get_logger

logger = get_logger(__name__)

class NematusFramework(Framework):

    def __init__(self):
        super(NematusFramework, self).__init__()
        self._nematus_dir = os.getenv('NEMATUS_DIR', '/root/nematus')

    def train(self,
              config,
              src_file,
              tgt_file,
              model_path=None,
              gpuid=0):
        options = _getTrainingOptions(config, gpuid=gpuid)
        options["datasets"] = src_file + " " + tgt_file

        src_vocab_file = src_file
        tgt_vocab_file = tgt_file
        if ("src_vocab" in options and
            "tgt_vocab" in options and
            os.path.isfile(options["src_vocab"]) and
            os.path.isfile(options["tgt_vocab"])):
            src_vocab_file = self._data_dir+"/src.vocab.txt"
            tgt_vocab_file = self._data_dir+"/tgt.vocab.txt"
            shutil.copy(options["src_vocab"], src_vocab_file)
            shutil.copy(options["tgt_vocab"], tgt_vocab_file)

        if "src_vocab" in options:
            del(options["src_vocab"])
        if "tgt_vocab" in options:
            del(options["tgt_vocab"])

        # generate dictionary
        self._run_command(None, ["python", "data/build_dictionary.py", src_vocab_file])
        self._run_command(None, ["python", "data/build_dictionary.py", tgt_vocab_file])
        options["dictionaries"] = " ".join([src_vocab_file + ".json", tgt_vocab_file + ".json"])

        if model_path is not None:
            options["reload"] == "true"
            options["model"] = model_path
        else:
            options["model"] = self._output_dir
        options["model"] += "/model.npz"

        env, options = _buildCommandLineOptions(options)

        self._run_command(env, ["python", "nematus/nmt.py"] + options)

        models = {}
        models["model.npz"] = self._output_dir + "model.npz"
        models["model.npz.gradinfo.npz"] = self._output_dir + "model.npz.gradinfo.npz"
        models["model.npz.json"] = self._output_dir + "model.npz.json"
        models["model.npz.progress.json"] = self._output_dir + "model.npz.progress.json"

        return models

    def trans(self, config, model_path, input, output, gpuid=0):
        model_file = os.path.join(model_path, 'model.npz')
        options = _getTranslationOptions(config, model_file, gpuid=gpuid)
        options['input'] = input
        options['output'] = output
        env, options = _buildCommandLineOptions(options)
        self._run_command(env, ["python", "nematus/translate.py"] + options)

    def _run_command(self, env, cmd):
        run_env = os.environ.copy()
        if env is not None:
            for k, v in six.iteritems(env):
                logger.debug("ENV %s", k + "=" + str(v))
                run_env[k] = str(v)

        logger.debug("RUN %s", " ".join(cmd))
        subprocess.call(shlex.split(" ".join(cmd)), cwd=self._nematus_dir, env=run_env)


def _getTheanoOptions(config, options=None, gpuid=0):
    assert options is None
    options = {}
    options_theano = {}

    if 'options' in config and isinstance(config['options'], dict):
        opt = config['options']
        # Theano options first
        if 'theano' in opt and isinstance(opt['theano'], dict):
            options_theano.update(opt['theano'])

    theano_value = ""
    for k, v in six.iteritems(options_theano):
        if k == "device":
            if gpuid == 0:
                theano_value += k + "=" + "cpu" + ","
            else:
                theano_value += k + "=" + "gpu" + ","
        elif k == "default":
            theano_value += v
        else:
            theano_value += k + "=" + v + ","
    theano_value = theano_value[:-1]

    options['theano'] = theano_value
    options['gpuid'] = gpuid
    return options

def _getTrainingOptions(config, options=None, gpuid=0):
    options = _getTheanoOptions(config, options=options, gpuid=gpuid)
    if 'options' in config and isinstance(config['options'], dict):
        opt = config['options']
        # Maybe overridden by training specific options
        if 'train' in opt and isinstance(opt['train'], dict):
            options.update(opt['train'])
    if not 'max_epochs' in options:
        options['max_epochs'] = 1
    return options

def _getTranslationOptions(config, model_t7, options=None, gpuid=0):
    options = _getTheanoOptions(config, options=options, gpuid=gpuid)
    if 'options' in config and isinstance(config['options'], dict):
        opt = config['options']
        # Maybe overridden by translation specific options
        if 'trans' in config and isinstance(config['trans'], dict):
            options.update(opt['trans'])
    # Misc "hard-coded" options
    options['models'] = model_t7
    return options

def _buildCommandLineOptions(options):
    env = {}
    opts = []
    for k, v in six.iteritems(options):
        if k == "theano":
            env["THEANO_FLAGS"] = v
        elif k == "gpuid":
            if v != 0:
                # to be consistent with LUA GPUID
                v -= 1
                env["CUDA_VISIBLE_DEVICES"] = v
        elif k is not None:
            if v != "false":
                opts.append('--%s' % k)
                if v != "true":
                    opts.append(str(v))
    return env, opts


if __name__ == "__main__":
    NematusFramework().run()
