import sys
import os
import pytest
import json
import functools

from nmtwizard.cloud_translation_framework import CloudTranslationFramework
from nmtwizard import serving


def _generate_numbers_file(path, max_count=12):
    with open(path, "w") as f:
        for i in range(max_count):
            f.write("%d\n" % i)
    return path

def _count_lines(path):
    with open(path, "rb") as f:
        i = 0
        for _ in f:
            i += 1
        return i

class _CopyTranslationFramework(CloudTranslationFramework):
    def translate_batch(self, batch, source_lang, target_lang):
        return batch

def _test_framework(tmpdir, framework_class):
    os.environ["WORKSPACE_DIR"] = str(tmpdir.join("workspace"))
    framework = framework_class()
    config = {"source": "en", "target": "fr"}
    input_path = str(tmpdir.join("input.txt"))
    output_path = str(tmpdir.join("output.txt"))
    _generate_numbers_file(input_path)
    args = [
        "-c", json.dumps(config),
        "trans",
        "-i", input_path,
        "-o", output_path,
    ]
    framework.run(args=args)
    assert os.path.isfile(output_path)
    assert _count_lines(input_path) == _count_lines(output_path)

def _test_real_framework(tmpdir, directory):
    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.insert(0, os.path.join(root_dir, "frameworks", directory))
    import entrypoint
    class_name = None
    for symbol in dir(entrypoint):
        if symbol.endswith("Framework") and symbol != "CloudTranslationFramework":
            class_name = symbol
    _test_framework(tmpdir, getattr(entrypoint, class_name))
    sys.path.pop(0)
    del sys.modules["entrypoint"]


def test_cloud_translation_framework(tmpdir):
    _test_framework(tmpdir, _CopyTranslationFramework)

def test_serve_cloud_translation_framework():
    class _ReverseTranslationFramework(CloudTranslationFramework):
        def translate_batch(self, batch, source_lang, target_lang):
            assert source_lang == "en"
            assert target_lang == "fr"
            return ["".join(reversed(list(text))) for text in batch]

    framework = _ReverseTranslationFramework()
    config = {"source": "en", "target": "fr"}
    _, service_info = framework.serve(config, None)
    request = {"src": [{"text": "Hello"}]}
    result = serving.run_request(
        request,
        framework._preprocess_input,
        functools.partial(framework.forward_request, service_info),
        framework._postprocess_output)
    _, service_info = framework.serve(config, None)
    assert result["tgt"][0][0]["text"] == "olleH"

@pytest.mark.skipif(
    "BAIDU_APPID" not in os.environ or "BAIDU_KEY" not in os.environ,
    reason="missing Baidu credentials")
def test_baidu_translate(tmpdir):
    _test_real_framework(tmpdir, "baidu_translate")

@pytest.mark.skipif(
    "DEEPL_CREDENTIALS" not in os.environ,
    reason="missing DeepL credentials")
def test_deepl_translate(tmpdir):
    _test_real_framework(tmpdir, "deepl_translate")

@pytest.mark.skipif(
    "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ,
    reason="missing Google credentials")
def test_google_translate(tmpdir):
    _test_real_framework(tmpdir, "google_translate")

@pytest.mark.skipif(
    "NAVER_CLIENT_ID" not in os.environ or "NAVER_SECRET" not in os.environ,
    reason="missing Naver credentials")
def test_naver_translate(tmpdir):
    _test_real_framework(tmpdir, "naver_translate")

@pytest.mark.skipif(
    "SOGOU_PID" not in os.environ or "SOGOU_KEY" not in os.environ,
    reason="missing Sogou credentials")
def test_sogou_translate(tmpdir):
    _test_real_framework(tmpdir, "sogou_translate")

@pytest.mark.skipif(
    "TENCENT_SecretId" not in os.environ or "TENCENT_SecretKey" not in os.environ,
    reason="missing Tencent credentials")
def test_tencent_translate(tmpdir):
    _test_real_framework(tmpdir, "tencent_translate")

@pytest.mark.skipif(
    "YOUDAO_APPID" not in os.environ or "YOUDAO_KEY" not in os.environ,
    reason="missing Youdao credentials")
def test_youdao_translate(tmpdir):
    _test_real_framework(tmpdir, "youdao_translate")
