from nmtwizard.framework import Framework
from nmtwizard.logger import get_logger
from nmtwizard.serving import TranslationOutput
import os
import re
import json
import urllib2

logger = get_logger(__name__)

supportedLangRe = re.compile(r"^(zh|en|fr|id|ko|es|th|zt|vi)$")
naver_lang_dict_map = {
  'zh':'zh-CN',
  'en':'en',
  'fr':'fr',
  'id':'id',
  'ko':'ko',
  'es':'es',
  'th':'th',
  'zt':'zh-TW',
  'vi':'vi'
}
entrypoint = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation"

class NaverTranslateFramework(Framework):

    def __init__(self):
        super(NaverTranslateFramework, self).__init__(stateless=True)
        self._appid = os.getenv('NAVER_CLIENT_ID')
        self._key = os.getenv('NAVER_SECRET')
        assert isinstance(self._appid, str), "missing app id"
        assert isinstance(self._key, str), "missing key"
        
    def trans(self, config, model_path, input, output, gpuid=0):
        assert supportedLangRe.match(config['source']), "unsupported language: %s" % config['source']
        assert supportedLangRe.match(config['target']), "unsupported language: %s" % config['target']
        with open(input, 'rb') as fi, open(output, 'wb') as fo:
            lines = fi.readlines()
            translations = translate_list(
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
            self._appid,
            self._key,
            batch_inputs,
            source_language=info['source'],
            target_language=info['target'])]

    def _preprocess_input(self, state, input, extra_config):
        return input

    def _postprocess_output(self, state, source, target, extra_config):
        return target


def translate_list(appid, secretKey, texts, source_language, target_language):

    i = 0
    while i < len(texts):
        nexti = i + 10
        if nexti > len(texts):
            nexti = len(texts)
        logger.info('Translating range [%d:%d]', i, nexti)
        
        query = ''.join(texts[i:nexti])
        fromLang = naver_lang_dict_map[source_language.lower()]
        toLang = naver_lang_dict_map[target_language.lower()]

        encText = urllib2.quote(query)
        data = "source="+fromLang+"&target="+toLang+"&text=" + encText

        request = urllib2.Request(entrypoint)
        request.add_header("X-NCP-APIGW-API-KEY-ID", appid)
        request.add_header("X-NCP-APIGW-API-KEY", secretKey)

        retry = 0
        while retry < 10:
            r = urllib2.urlopen(request, data=data.encode("utf-8"))
            rescode = r.getcode()
            if rescode == 429:
                retry += 1
                time.sleep(5)
            else:
                break

        results = json.load(r)
        if rescode != 200 or 'result' not in results['message']:
            raise RuntimeError('incorrect result from \'translate\' service: %s' % results)
        yield results['message']['result']['translatedText']
        i = nexti

if __name__ == "__main__":
    NaverTranslateFramework().run()
