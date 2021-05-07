import os
import hashlib
import random
import requests

import six

from nmtwizard.cloud_translation_framework import CloudTranslationFramework


youdao_lang_dict_map = {
    "zh": "zh-CHS",
    "en": "en",
    "fr": "fr",
    "ja": "ja",
    "ko": "ko",
    "pt": "pt",
    "ru": "ru",
    "es": "es",
    "vi": "vi",
}


class YoudaoTranslateFramework(CloudTranslationFramework):
    def __init__(self):
        super(YoudaoTranslateFramework, self).__init__()
        self._appid = os.getenv("YOUDAO_APPID")
        self._key = os.getenv("YOUDAO_KEY")
        if self._appid is None:
            raise ValueError("missing app id")
        if self._key is None:
            raise ValueError("missing key")

    def translate_batch(self, batch, source_lang, target_lang):
        query = "\n".join(batch)
        salt = str(random.randint(10000, 99999))
        sign = self._appid + query + salt + self._key
        m1 = hashlib.md5()
        m1.update(six.b(sign))
        sign = m1.hexdigest()

        url = "http://openapi.youdao.com/api"
        params = {
            "appKey": self._appid,
            "q": query,
            "from": youdao_lang_dict_map[source_lang.lower()],
            "to": youdao_lang_dict_map[target_lang.lower()],
            "salt": salt,
            "sign": sign,
        }

        result = self.send_request(lambda: requests.get(url, params=params))
        for trans in result["translation"]:
            yield trans

    def supported_languages(self):
        return ["zh", "en", "fr", "ja", "ko", "pt", "ru", "es", "vi"]


if __name__ == "__main__":
    YoudaoTranslateFramework().run()
