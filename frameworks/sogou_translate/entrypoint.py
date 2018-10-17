from nmtwizard.framework import Framework
from nmtwizard.logger import get_logger
from nmtwizard.serving import TranslationOutput
import os
import re
import json
import time
import httplib
import urllib
import random
import hashlib

logger = get_logger(__name__)

supportedLangRe = re.compile(r"^(ar|bn|bg|zh|hr|cs|da|nl|en|et|fi|fr|de|el|he|hi|hu|id|it|ja|ko|lv|lt|ms|no|fa|pl|pt|ro|ru|sr|sb|sk|sl|es|sv|th|zt|tr|uk|ur|vi|cy)$")
sogou_lang_dict_map = {
  'ar':'ar',
  'bn':'bn',
  'bg':'bg',
  'zh':'zh-CHS',
  'hr':'hr',
  'cs':'cs',
  'da':'da',
  'nl':'nl',
  'en':'en',
  'et':'et',
  'fi':'fil',
  'fr':'fr',
  'de':'de',
  'el':'el',
  'he':'he',
  'hi':'hi',
  'hu':'hu',
  'id':'id',
  'it':'it',
  'ja':'ja',
  'ko':'ko',
  'lv':'lv',
  'lt':'lt',
  'ms':'ms',
  'no':'no',
  'fa':'fa',
  'pl':'pl',
  'pt':'pt',
  'ro':'ro',
  'ru':'ru',
  'sr':'sr-Cyrl',
  'sb':'sr-Latn',
  'sk':'sk',
  'sl':'sl',
  'es':'es',
  'sv':'sv',
  'th':'th',
  'zt':'zh-CHT',
  'tr':'tr',
  'uk':'uk',
  'ur':'ur',
  'vi':'vi',
  'cy':'cy'
}
entrypoint = "fanyi.sogou.com:80"

class SogouTranslateFramework(Framework):

    def __init__(self):
        super(SogouTranslateFramework, self).__init__(stateless=True)
        self._appid = os.getenv('SOGOU_PID')
        self._key = os.getenv('SOGOU_KEY')
        assert isinstance(self._appid, str), "missing pid"
        assert isinstance(self._key, str), "missing key"
        self.httpClient = httplib.HTTPConnection(entrypoint)
        
    def __del__(self):
        if self.httpClient:
          self.httpClient.close()
          
    def trans(self, config, model_path, input, output, gpuid=0):
        assert supportedLangRe.match(config['source']), "unsupported language: %s" % config['source']
        assert supportedLangRe.match(config['target']), "unsupported language: %s" % config['target']
        with open(input, 'rb') as fi, open(output, 'wb') as fo:
            lines = fi.readlines()
            translations = translate_list(
                self.httpClient,
                self._appid,
                self._key,
                lines, source_language=config['source'], target_language=config['target'])
            for translation in translations:
                fo.write(translation.encode('utf-8') + '\n')

    def train(self, *args, **kwargs):
        raise NotImplementedError("This framework can only be used for translation")

    def release(self, *arg, **kwargs):
        raise NotImplementedError('This framework does not require a release step')

    def serve(self, config, model_path, gpuid=0):
        return None, {'source': config['source'], 'target': config['target']}

    def forward_request(self, batch_inputs, info, timeout=None):
        return [[TranslationOutput(translation)] for translation in translate_list(
            self.httpClient,
            self._appid,
            self._key,
            batch_inputs,
            source_language=info['source'],
            target_language=info['target'])]

    def _preprocess_input(self, state, input, extra_config):
        return input

    def _postprocess_output(self, state, source, target, extra_config):
        return target


def translate_list(httpClient, appid, secretKey, texts, source_language, target_language):

    i = 0
    while i < len(texts):
        nexti = i + 10
        if nexti > len(texts):
            nexti = len(texts)
        logger.info('Translating range [%d:%d]', i, nexti)
        
        query = ''.join(texts[i:nexti])
        fromLang = sogou_lang_dict_map[source_language.lower()]
        toLang = sogou_lang_dict_map[target_language.lower()]
        salt = random.randint(10000, 99999)

        sign = appid+query.strip()+str(salt)+secretKey
        sign = hashlib.md5(sign.encode('utf-8')).hexdigest()

        payload = 'from='+fromLang+'&to='+toLang+'&pid='+appid+'&q='+urllib.quote(query)+'&sign='+sign+'&salt='+str(salt)

        headers = {
            'content-type': "application/x-www-form-urlencoded",
            'accept': "application/json"
            }        

        retry = 0
        while retry < 10:
            httpClient.request("POST", "/reventondc/api/sogouTranslate", payload, headers)
            r = httpClient.getresponse()
            if r.status == 429:
                retry += 1
                time.sleep(5)
            else:
                break

        results = json.load(r)
        if r.status != 200 or 'translation' not in results:
            raise RuntimeError('incorrect result from \'translate\' service: %s' % results)
        yield results['translation']
        i = nexti

if __name__ == "__main__":
    SogouTranslateFramework().run()
