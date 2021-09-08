import os
import hashlib
import random
import requests

from nmtwizard.cloud_translation_framework import CloudTranslationFramework


baidu_lang_dict_map = {
    "ar": "ara",
    "bg": "bul",
    "zh": "zh",
    "cs": "cs",
    "da": "dan",
    "nl": "nl",
    "en": "en",
    "et": "est",
    "fi": "fin",
    "fr": "fra",
    "de": "de",
    "el": "el",
    "hu": "hu",
    "it": "it",
    "ja": "jp",
    "ko": "kor",
    "pl": "pl",
    "pt": "pt",
    "ro": "rom",
    "ru": "ru",
    "sl": "slo",
    "es": "spa",
    "sv": "swe",
    "th": "th",
    "zt": "cht",
    "vi": "vie",
}


class BaiduTranslateFramework(CloudTranslationFramework):
    def __init__(self):
        super(BaiduTranslateFramework, self).__init__()
        self._appid = os.getenv("BAIDU_APPID")
        self._key = os.getenv("BAIDU_KEY")
        if self._appid is None:
            raise ValueError("missing app id")
        if self._key is None:
            raise ValueError("missing key")

    def translate_batch(self, batch, source_lang, target_lang):
        query = "\n".join(batch)
        salt = str(random.randint(10000, 99999))
        sign = self._appid + query + salt + self._key
        m1 = hashlib.md5()
        m1.update(sign.encode("utf-8"))
        sign = m1.hexdigest()

        params = {
            "appid": self._appid,
            "q": query,
            "from": baidu_lang_dict_map[source_lang.lower()],
            "to": baidu_lang_dict_map[target_lang.lower()],
            "salt": salt,
            "sign": sign,
        }

        url = "http://api.fanyi.baidu.com/api/trans/vip/translate"
        result = self.send_request(lambda: requests.get(url, params=params))
        for trans in result["trans_result"]:
            yield trans["dst"]

    def supported_languages(self):
        return [
            "ar",
            "bg",
            "cs",
            "da",
            "de",
            "el",
            "en",
            "es",
            "et",
            "fi",
            "fr",
            "hu",
            "it",
            "ja",
            "ko",
            "nl",
            "pl",
            "pt",
            "ro",
            "ru",
            "sl",
            "sv",
            "th",
            "vi",
            "zh",
            "zt",
        ]


if __name__ == "__main__":
    BaiduTranslateFramework().run()
