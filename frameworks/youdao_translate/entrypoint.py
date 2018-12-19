import os
import json
import time
import httplib
import md5
import urllib
import random

from nmtwizard.cloud_translation_framework import CloudTranslationFramework


youdao_lang_dict_map = {
  'zh':'zh-CHS',
  'en':'en',
  'fr':'fr',
  'ja':'ja',
  'ko':'ko',
  'pt':'pt',
  'ru':'ru',
  'es':'es',
  'vi':'vi'
}


class YoudaoTranslateFramework(CloudTranslationFramework):

    def __init__(self):
        super(YoudaoTranslateFramework, self).__init__()
        self._appid = os.getenv('YOUDAO_APPID')
        self._key = os.getenv('YOUDAO_KEY')
        if self._appid is None:
            raise ValueError("missing app id")
        if self._key is None:
            raise ValueError("missing key")
        self.httpClient = httplib.HTTPConnection("openapi.youdao.com")

    def __del__(self):
        if self.httpClient:
          self.httpClient.close()

    def translate_batch(self, batch, source_lang, target_lang):
        query = '\n'.join(batch)
        from_lang = youdao_lang_dict_map[source_lang.lower()]
        to_lang = youdao_lang_dict_map[target_lang.lower()]
        salt = str(random.randint(10000, 99999))

        sign = self._appid + query + salt + self._key
        m1 = md5.new()
        m1.update(sign)
        sign = m1.hexdigest()

        myurl = '/api?appKey=%s&q=%s&from=%s&to=%s&salt=%s&sign=%s' % (
            self._appid, urllib.quote(query), from_lang, to_lang, salt, sign)

        retry = 0
        while retry < 10:
            self.httpClient.request('GET', myurl)
            r = self.httpClient.getresponse()
            if r.status == 429:
                retry += 1
                time.sleep(5)
            else:
                break

        results = json.load(r)
        if r.status != 200 or 'translation' not in results:
            raise RuntimeError('incorrect result from \'translate\' service: %s' % results)
        for trans in results['translation']:
            yield trans

    def supported_languages(self):
        return ['zh', 'en', 'fr', 'ja', 'ko', 'pt', 'ru', 'es', 'vi']


if __name__ == "__main__":
    YoudaoTranslateFramework().run()
