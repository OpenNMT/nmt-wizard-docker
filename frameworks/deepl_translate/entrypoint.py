from nmtwizard.framework import Framework
from nmtwizard.logger import get_logger
from nmtwizard.serving import TranslationOutput
import requests
import os
import re
import json
import time

logger = get_logger(__name__)

supportedLangRe = re.compile(r"^(en|de|fr|es|it|nl|pl)$")
entrypoint = "https://api.deepl.com/v1/translate"

class DeepLTranslateFramework(Framework):

    def __init__(self):
        super(DeepLTranslateFramework, self).__init__(stateless=True)
        self._credentials = os.getenv('DEEPL_CREDENTIALS')
        assert isinstance(self._credentials, str), "missing credentials"

    def trans(self, config, model_path, input, output, gpuid=0):
        assert supportedLangRe.match(config['source']), "unsupported language: %s" % config['source']
        assert supportedLangRe.match(config['target']), "unsupported language: %s" % config['target']
        with open(input, 'rb') as fi, open(output, 'wb') as fo:
            lines = fi.readlines()
            translations = translate_list(
                self._credentials,
                lines, source_language=config['source'], target_language=config['target'])
            for translation in translations:
                fo.write(translation.encode('utf-8'))

    def train(self, *args, **kwargs):
        raise NotImplementedError("This framework can only be used for translation")

    def release(self, *arg, **kwargs):
        raise NotImplementedError('This framework does not require a release step')

    def serve(self, config, model_path, gpuid=0):
        return None, {'source': config['source'], 'target': config['target']}

    def forward_request(self, batch_inputs, info, timeout=None):
        return [[TranslationOutput(translation)] for translation in translate_list(
            self._credentials,
            batch_inputs,
            source_language=info['source'],
            target_language=info['target'])]

    def _preprocess_input(self, state, input):
        return input

    def _postprocess_output(self, state, output):
        return output


def translate_list(credentials, texts, source_language, target_language):

    i = 0
    while i < len(texts):
        nexti = i + 10
        if nexti > len(texts):
            nexti = len(texts)
        logger.info('Translating range [%d:%d]', i, nexti)
        params = { 
          "text": texts[i:nexti],
          "source_lang": source_language.upper(),
          "target_lang": target_language.upper(),
          "split_sentences": 0,
          "auth_key": credentials
        }

        retry = 0
        while retry < 10:
            r = requests.get(entrypoint, params=params)
            if r.status_code == 429:
                retry += 1
                time.sleep(5)
            else:
                break

        if r.status_code != 200 or 'translations' not in r.json():
            raise RuntimeError('incorrect result from \'translate\' service: %s' % r.text)
        for trans in r.json()['translations']:
            yield trans['text']
        i = nexti

if __name__ == "__main__":
    DeepLTranslateFramework().run()
