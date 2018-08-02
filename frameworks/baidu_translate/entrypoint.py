from nmtwizard.framework import Framework
from nmtwizard.logger import get_logger
from nmtwizard.serving import TranslationOutput
import os
import re
import json
import time
import httplib
import md5
import urllib
import random

logger = get_logger(__name__)

supportedLangRe = re.compile(r"^(ar|bg|zh|cs|da|nl|en|et|fi|fr|de|el|hu|it|ja|ko|pl|pt|ro|ru|sl|es|sv|th|zt|vi)$")
baidu_lang_dict_map = {
  'ar':'ara',
  'bg':'bul',
  'zh':'zh',
  'cs':'cs',
  'da':'dan',
  'nl':'nl',
  'en':'en',
  'et':'est',
  'fi':'fin',
  'fr':'fra',
  'de':'de',
  'el':'el',
  'hu':'hu',
  'it':'it',
  'ja':'jp',
  'ko':'kor',
  'pl':'pl',
  'pt':'pt',
  'ro':'rom',
  'ru':'ru',
  'sl':'slo',
  'es':'spa',
  'sv':'swe',
  'th':'th',
  'zt':'cht',
  'vi':'vie'
}
entrypoint = "fanyi-api.baidu.com"

class BaiduTranslateFramework(Framework):

    def __init__(self):
        super(BaiduTranslateFramework, self).__init__(stateless=True)
        self._appid = os.getenv('BAIDU_APPID')
        self._key = os.getenv('BAIDU_KEY')
        assert isinstance(self._appid, str), "missing app id"
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

    def _preprocess_input(self, state, input):
        return input

    def _postprocess_output(self, state, output):
        return output


def translate_list(httpClient, appid, secretKey, texts, source_language, target_language):

    i = 0
    while i < len(texts):
        nexti = i + 10
        if nexti > len(texts):
            nexti = len(texts)
        logger.info('Translating range [%d:%d]', i, nexti)
        
        query = ''.join(texts[i:nexti])
        fromLang = baidu_lang_dict_map[source_language.lower()]
        toLang = baidu_lang_dict_map[target_language.lower()]
        salt = random.randint(10000, 99999)

        sign = appid+query+str(salt)+secretKey
        m1 = md5.new()
        m1.update(sign)
        sign = m1.hexdigest()

        myurl = '/api/trans/vip/translate'+'?appid='+appid+'&q='+urllib.quote(query)+'&from='+fromLang+'&to='+toLang+'&salt='+str(salt)+'&sign='+sign
        
        retry = 0
        while retry < 10:
            httpClient.request('GET', myurl)
            r = httpClient.getresponse()
            if r.status == 429:
                retry += 1
                time.sleep(5)
            else:
                break

        results = json.load(r)
        if r.status != 200 or 'trans_result' not in results:
            raise RuntimeError('incorrect result from \'translate\' service: %s' % results)
        for trans in results['trans_result']:
            yield trans['dst']
        i = nexti

if __name__ == "__main__":
    BaiduTranslateFramework().run()
