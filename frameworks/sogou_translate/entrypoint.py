import os
import random
import hashlib
import requests

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

    def translate_batch(self, batch, source_lang, target_lang):
        query = '\n'.join(batch)
        salt = str(random.randint(10000, 99999))
        sign = self._appid + query + salt + self._key
        sign = hashlib.md5(sign.encode('utf-8')).hexdigest()

        url = 'http://fanyi.sogou.com:80/reventondc/api/sogouTranslate'
        data = {
            'from': sogou_lang_dict_map[source_lang.lower()],
            'to': sogou_lang_dict_map[target_lang.lower()],
            'pid': self._appid,
            'q': query,
            'sign': sign,
            'salt': salt
        }
        headers = {
            'content-type': "application/x-www-form-urlencoded",
            'accept': "application/json"
        }

        result = self.send_request(lambda: requests.post(url, data=data, headers=headers))
        yield result['translation']

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
