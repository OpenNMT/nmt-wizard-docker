import collections
import json
import functools
import logging
import signal
import threading
import copy
import socket
import time
import six

from six.moves import SimpleHTTPServer
from six.moves import socketserver

from nmtwizard import config as config_util
from nmtwizard.logger import get_logger

logger = get_logger(__name__)


class TranslationOutput(object):
    """Simple structure holding translation outputs."""
    def __init__(self, output, score=None, attention=None):
        self.output = output
        self.score = score
        self.attention = attention

class TranslationExample(
        collections.namedtuple("TranslationExample",
                               ("config", "tokens", "metadata"))):

    @property
    def num_parts(self):
        return len(self.tokens)

def pick_free_port():
    """Selects an available port."""
    s = socket.socket()
    s.bind(('', 0))
    return s.getsockname()[1]

def start_server(host,
                 port,
                 config,
                 serving_state,
                 backend_service_fn,
                 preprocess_fn,
                 translate_fn,
                 postprocess_fn):
    """Start a serving service.

    This function will only return on SIGINT or SIGTERM signals.

    Args:
      host: The hostname of the service.
      port: The port used by the service.
      serving_state: The framework state to propagate to pre/postprocessing callbacks.
      backend_service_fn: A callable to start the framework dependent backend service.
      preprocess_fn: A callable taking (serving_state, text, config) and returning tokens.
      translation_fn: A callable that forwards the request to the translation backend.
      postprocess_fn: A callable taking (serving_state, src_tokens, tgt_tokens, config)
        and returning text.
    """
    global backend_process
    global backend_info
    backend_process, backend_info = backend_service_fn()
    global_timeout = None
    global_max_batch_size = None
    serving_config = config.get('serving')
    if serving_config is not None and isinstance(serving_config, dict):
        global_timeout = config.get('timeout')
        global_max_batch_size = config.get('max_batch_size')

    class ServerHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
        def do_POST(self):
            if self.path == '/translate':
                self.translate()
            elif self.path == '/unload_model':
                self.unload_model()
            elif self.path == '/reload_model':
                self.reload_model()
            else:
                self.send_error(404, 'invalid route %s' % self.path)

        def translate(self):
            global backend_process
            if backend_process is not None and not _process_is_running(backend_process):
                self.send_error(503, 'backend service is unavailable')
                return
            header_fn = (
                self.headers.getheader if hasattr(self.headers, "getheader")
                else self.headers.get)
            content_len = int(header_fn('content-length', 0))
            if content_len == 0:
                self.send_error(400, 'missing request data')
                return
            post_body = self.rfile.read(content_len)
            try:
                result = run_request(
                    json.loads(six.ensure_str(post_body)),
                    functools.partial(preprocess_fn, serving_state),
                    functools.partial(translate_fn, info=backend_info),
                    functools.partial(postprocess_fn, serving_state),
                    config=config,
                    max_batch_size=global_max_batch_size,
                    timeout=global_timeout)
            except ValueError as e:
                self.send_error(400, str(e))
            except RuntimeError as e:
                self.send_error(504, str(e))
            else:
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(six.ensure_binary(json.dumps(result)))

        def unload_model(self):
            global backend_process
            global backend_info
            if backend_process is not None and _process_is_running(backend_process):
                backend_process.terminate()
            backend_process = None
            backend_info = None
            self.send_response(200)

        def reload_model(self):
            global backend_process
            global backend_info
            if backend_process is not None and _process_is_running(backend_process):
                backend_process.terminate()
            backend_process, backend_info = backend_service_fn()
            self.send_response(200)

    try:
        frontend_server = socketserver.ThreadingTCPServer((host, port), ServerHandler)
    except socket.error as e:
        if backend_process is not None:
            backend_process.terminate()
        raise e

    def shutdown(signum, frame):
        frontend_server.shutdown()
        if backend_process is not None:
            backend_process.terminate()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info('Serving model on port %d', port)
    server_thread = threading.Thread(target=frontend_server.serve_forever)
    server_thread.start()
    while server_thread.is_alive():
        time.sleep(1)
    frontend_server.server_close()


def run_request(request,
                preprocess_fn,
                translate_fn,
                postprocess_fn,
                config=None,
                max_batch_size=None,
                timeout=None):
    """Runs a translation request."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Incoming request: %s", json.dumps(request, ensure_ascii=False))

    if not isinstance(request, dict):
        raise ValueError('request should be a JSON object')
    src = request.get('src')
    if src is None:
        raise ValueError('missing src field')
    if not isinstance(src, list):
        raise ValueError('src field must be a list')

    if not src:
        results = []
    else:
        # Read request specific options and config.
        options = request.get('options', {})
        config = finalize_config(config, override=options.get('config'))
        max_batch_size = options.get('max_batch_size', max_batch_size)
        timeout = options.get('timeout', timeout)

        examples = preprocess_examples(src, preprocess_fn, config=config)
        outputs = translate_examples(
            examples,
            translate_fn,
            max_batch_size=max_batch_size,
            timeout=timeout)
        results = postprocess_outputs(outputs, examples, postprocess_fn)

    return {'tgt': results}

def finalize_config(config, override=None, options=None):
    """Finalizes the configuration with possible override and options."""
    if config is not None and (override or options):
        config = copy.deepcopy(config)
        if override:
            config_util.merge_config(config, override)
        if options:
            config_util.update_config_with_options(config, options)
    return config

def preprocess_text(text, func, config=None):
    """Applies preprocessing function on text."""
    output = func(text, config)

    # Preprocessing may return additional metadata alongside tokens.
    if isinstance(output, tuple):
        tokens, metadata = output
    else:
        tokens, metadata = output, None

    # Move to the general multiparts representation.
    if not tokens or not isinstance(tokens[0], list):
        tokens = [tokens]
        metadata = [metadata]

    return TranslationExample(
        config=config,
        tokens=tokens,
        metadata=metadata)

def preprocess_examples(raw_examples, func, config=None):
    """Applies preprocessing on a list of example structures."""
    examples = []
    for i, raw_example in enumerate(raw_examples):
        if not isinstance(raw_example, dict):
            raise ValueError('example %d is not a JSON object' % i)
        text = raw_example.get('text')
        if text is None:
            raise ValueError('missing text field in example %d' % i)
        example_config = finalize_config(
            config,
            override=raw_example.get('config'),
            options=raw_example.get('options'))
        example = preprocess_text(text, func, config=example_config)
        examples.append(example)
    return examples

def postprocess_output(output, example, func):
    """Applies postprocessing function on a translation output."""
    if example.num_parts > 1:
        # For multi parts inputs, send all parts to the postprocessing.
        tgt_tokens = output.output
        src_context = (example.tokens, example.metadata)
        score = sum(output.score) if all(s is not None for s in output.score) else None
        align = None
    else:
        # Otherwise just take the first element and pass metadata only if defined.
        tgt_tokens = output.output[0]
        src_tokens = example.tokens[0]
        src_metadata = example.metadata[0]
        src_context = src_tokens if src_metadata is None else (src_tokens, src_metadata)
        score = output.score[0]
        attention = output.attention[0]
        align = align_tokens(src_tokens, tgt_tokens, attention) if attention else None

    text = func(src_context, tgt_tokens, example.config)
    result = {'text': text}
    if score is not None:
        result['score'] = score
    if align is not None:
        result['align'] = align
    return result

def postprocess_outputs(outputs, examples, func):
    """Applies postprocess on model outputs."""
    results = []
    for hypotheses, example in zip(outputs, examples):
        results.append([
            postprocess_output(hypothesis, example, func)
            for hypothesis in hypotheses])
    return results

def align_tokens(src_tokens, tgt_tokens, attention):
    if not src_tokens or not tgt_tokens:
        return []
    src_ranges = []
    offset = 0
    for src_token in src_tokens:
        src_ranges.append((offset, offset + len(src_token)))
        offset += len(src_token) + 1
    alignments = []
    offset = 0
    for tgt_id, (tgt_token, attn) in enumerate(zip(tgt_tokens, attention)):
        src_id = attn.index(max(attn))
        tgt_range = (offset, offset + len(tgt_token))
        src_range = src_ranges[src_id]
        offset += len(tgt_token) + 1
        alignments.append({
            "tgt": [{"range": tgt_range, "id": tgt_id}],
            "src": [{"range": src_range, "id": src_id}]})
    return alignments

def translate_examples(examples, func, max_batch_size=None, timeout=None):
    """Translates examples."""
    flat_outputs = []
    for batch in batch_iterator(examples, max_batch_size=max_batch_size):
        hypotheses = func(batch, timeout=timeout)
        if hypotheses is None:
            raise RuntimeError('translation request timed out')
        flat_outputs.extend(hypotheses)
    return unflatten_outputs(flat_outputs, examples)

def batch_iterator(examples, max_batch_size=None):
    """Yields batch of tokens not larger than max_batch_size."""
    all_tokens = []
    for example in examples:
        all_tokens.extend(example.tokens)
    batch_size = len(all_tokens)
    if max_batch_size is None or batch_size <= max_batch_size:
        yield all_tokens
    else:
        offset = 0
        while offset < batch_size:
            lower_bound = offset
            upper_bound = min(offset + max_batch_size, batch_size)
            yield all_tokens[lower_bound:upper_bound]
            offset = upper_bound

def unflatten_outputs(flat_outputs, examples):
    """Unflattens outputs according to examples parts."""
    outputs = []
    offset = 0
    for example in examples:
        num_parts = example.num_parts
        hypotheses = flat_outputs[offset:offset + num_parts]
        offset += num_parts
        # Extract and merge parts of each hypothesis.
        num_hypotheses = len(hypotheses[0])
        outputs.append([
            merge_translation_outputs([part[h] for part in hypotheses])
            for h in range(num_hypotheses)])
    return outputs

def merge_translation_outputs(parts):
    """Merges multiple translation outputs in a single one."""
    output = []
    score = []
    attention = []
    for part in parts:
        output.append(part.output)
        score.append(part.score)
        attention.append(part.attention)
    return TranslationOutput(output, score=score, attention=attention)

def _process_is_running(process):
    return process is not None and process.poll() is None
