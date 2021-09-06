import os
import time
import hashlib
import hmac
import base64
import random
import sys
import binascii
import requests
import urllib.parse

from nmtwizard.cloud_translation_framework import CloudTranslationFramework


class TencentTranslateFramework(CloudTranslationFramework):
    def __init__(self):
        super(TencentTranslateFramework, self).__init__()
        self._appid = os.getenv("TENCENT_SecretId")
        self._key = os.getenv("TENCENT_SecretKey")
        if self._appid is None:
            raise ValueError("missing app id")
        if self._key is None:
            raise ValueError("missing key")

    def translate_batch(self, batch, source_lang, target_lang):
        # Tencent API does not support translating multi lines in one request
        for line in batch:
            yield self._translate_line(line, source_lang, target_lang)

    def _translate_line(self, line, source_lang, target_lang):
        url = "tmt.na-siliconvalley.tencentcloudapi.com"
        signature_method = "HmacSHA256"
        params = [
            ("Action", "TextTranslate"),
            ("Nonce", random.randint(1, sys.maxsize)),
            ("ProjectId", 0),
            ("Region", "na-siliconvalley"),
            ("SecretId", self._appid),
            ("SignatureMethod", signature_method),
            ("Source", source_lang.lower()),
            ("SourceText", line),
            ("Target", target_lang.lower()),
            ("Timestamp", int(time.time())),
            ("Version", "2018-03-21"),
        ]
        request = "GET%s/?%s" % (url, urllib.parse.urlencode(params))
        params.append(
            ("Signature", _sign_request(self._key, request, signature_method))
        )
        headers = {
            "content-type": "application/x-www-form-urlencoded",
            "accept": "application/json",
        }

        result = self.send_request(
            lambda: requests.get("https://" + url, params=params, headers=headers)
        )
        return result["Response"]["TargetText"]

    def supported_languages(self):
        return [
            "de",
            "en",
            "es",
            "fr",
            "id",
            "it",
            "ja",
            "ko",
            "ms",
            "pt",
            "ru",
            "th",
            "tr",
            "vi",
            "zh",
        ]


def _sign_request(secretKey, signStr, signMethod):
    signStr = bytes(signStr, "utf-8")
    secretKey = bytes(secretKey, "utf-8")

    digestmod = None
    if signMethod == "HmacSHA256":
        digestmod = hashlib.sha256
    elif signMethod == "HmacSHA1":
        digestmod = hashlib.sha1
    else:
        raise NotImplementedError(
            "signMethod invalid", "signMethod only support (HmacSHA1, HmacSHA256)"
        )

    hashed = hmac.new(secretKey, signStr, digestmod)
    return binascii.b2a_base64(hashed.digest())[:-1].decode()


if __name__ == "__main__":
    TencentTranslateFramework().run()
