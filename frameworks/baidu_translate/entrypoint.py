import os
import json
import time
import httplib
import md5
import urllib
import random

from nmtwizard.cloud_translation_framework import CloudTranslationFramework


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


class BaiduTranslateFramework(CloudTranslationFramework):

    def __init__(self):
        super(BaiduTranslateFramework, self).__init__()
        self._appid = os.getenv('BAIDU_APPID')
        self._key = os.getenv('BAIDU_KEY')
        if self._appid is None:
            raise ValueError("missing app id")
        if self._key is None:
            raise ValueError("missing key")
        self.httpClient = httplib.HTTPConnection("fanyi-api.baidu.com")

    def __del__(self):
        if self.httpClient:
          self.httpClient.close()

    def translate_batch(self, batch, source_lang, target_lang):
        query = '\n'.join(batch)
        from_lang = baidu_lang_dict_map[source_lang.lower()]
        to_lang = baidu_lang_dict_map[target_lang.lower()]
        salt = str(random.randint(10000, 99999))

        sign = self._appid + query + salt + self._key
        m1 = md5.new()
        m1.update(sign)
        sign = m1.hexdigest()

        url = '/api/trans/vip/translate?appid=%s&q=%s&from=%s&to=%s&salt=%s&sign=%s' % (
            self._appid, urllib.quote(query), from_lang, to_lang, salt, sign)

        retry = 0
        while retry < 10:
            self.httpClient.request('GET', url)
            r = self.httpClient.getresponse()
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

    def supported_languages(self):
        return [
            'ar',
            'bg',
            'cs',
            'da',
            'de',
            'el',
            'en',
            'es',
            'et',
            'fi',
            'fr',
            'hu',
            'it',
            'ja',
            'ko',
            'nl',
            'pl',
            'pt',
            'ro',
            'ru',
            'sl',
            'sv',
            'th',
            'vi',
            'zh',
            'zt'
        ]

if __name__ == "__main__":
    BaiduTranslateFramework().run()
