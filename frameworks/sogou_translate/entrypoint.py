import os
import json
import time
import httplib
import urllib
import random
import hashlib

from nmtwizard.cloud_translation_framework import CloudTranslationFramework


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


class SogouTranslateFramework(CloudTranslationFramework):

    def __init__(self):
        super(SogouTranslateFramework, self).__init__()
        self._appid = os.getenv('SOGOU_PID')
        self._key = os.getenv('SOGOU_KEY')
        if self._appid is None:
            raise ValueError("missing pid")
        if self._key is None:
            raise ValueError("missing key")
        self.httpClient = httplib.HTTPConnection("fanyi.sogou.com:80")

    def __del__(self):
        if self.httpClient:
          self.httpClient.close()

    def translate_batch(self, batch, source_lang, target_lang):
        query = '\n'.join(batch)
        from_lang = sogou_lang_dict_map[source_lang.lower()]
        to_lang = sogou_lang_dict_map[target_lang.lower()]
        salt = str(random.randint(10000, 99999))

        sign = self._appid + query.strip() + salt + self._key
        sign = hashlib.md5(sign.encode('utf-8')).hexdigest()

        payload = 'from=%s&to=%s&pid=%s&q=%s&sign=%s&salt=%s' % (
            from_lang, to_lang, self._appid, urllib.quote(query), sign, salt)

        headers = {
            'content-type': "application/x-www-form-urlencoded",
            'accept': "application/json"
        }

        retry = 0
        while retry < 10:
            self.httpClient.request("POST", "/reventondc/api/sogouTranslate", payload, headers)
            r = self.httpClient.getresponse()
            if r.status == 429:
                retry += 1
                time.sleep(5)
            else:
                break

        results = json.load(r)
        if r.status != 200 or 'translation' not in results:
            raise RuntimeError('incorrect result from \'translate\' service: %s' % results)
        yield results['translation']

    def supported_languages(self):
        return [
            'ar',
            'bg',
            'bn',
            'cs',
            'cy',
            'da',
            'de',
            'el',
            'en',
            'es',
            'et',
            'fa',
            'fi',
            'fr',
            'he',
            'hi',
            'hr',
            'hu',
            'id',
            'it',
            'ja',
            'ko',
            'lt',
            'lv',
            'ms',
            'nl',
            'no',
            'pl',
            'pt',
            'ro',
            'ru',
            'sb',
            'sk',
            'sl',
            'sr',
            'sv',
            'th',
            'tr',
            'uk',
            'ur',
            'vi',
            'zh',
            'zt'
        ]


if __name__ == "__main__":
    SogouTranslateFramework().run()
