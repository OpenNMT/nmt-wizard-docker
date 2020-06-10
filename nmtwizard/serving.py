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
                               (
                                   "index",
                                   "config",
                                   "source_tokens",
                                   "target_tokens",
                                   "mode",
                                   "metadata",
                               ))):

    @property
    def num_parts(self):
        return len(self.source_tokens)

class TranslationBatch(
        collections.namedtuple("TranslationBatch",
                               (
                                   "indices",
                                   "source_tokens",
                                   "target_tokens",
                                   "mode",
                               ))):
    pass


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
        def _send_response(self, data):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(six.ensure_binary(json.dumps(data)))

        def do_GET(self):
            if self.path == '/status':
                self.status()
            else:
                self.send_error(404, 'invalid route %s' % self.path)

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
            if (backend_info is None or
                (backend_process is not None and not _process_is_running(backend_process))):
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
                    functools.partial(translate_fn, backend_info),
                    functools.partial(postprocess_fn, serving_state),
                    config=config,
                    max_batch_size=global_max_batch_size,
                    timeout=global_timeout)
            except ValueError as e:
                self.send_error(400, str(e))
            except RuntimeError as e:
                self.send_error(504, str(e))
            else:
                self._send_response(result)

        def status(self):
            if backend_info is None:
                status = "unloaded"
            else:
                status = "ready"
            self._send_response({"status": status})

        def unload_model(self):
            global backend_process
            global backend_info
            if backend_process is not None and _process_is_running(backend_process):
                backend_process.terminate()
            backend_process = None
            backend_info = None
            self.status()

        def reload_model(self):
            global backend_process
            global backend_info
            if backend_process is not None and _process_is_running(backend_process):
                backend_process.terminate()
            backend_process, backend_info = backend_service_fn()
            self.status()

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
        options.setdefault('timeout', timeout)
        config = finalize_config(config, override=options.get('config'))
        max_batch_size = options.get('max_batch_size', max_batch_size)

        examples = preprocess_examples(src, preprocess_fn, config=config)
        outputs = translate_examples(
            examples,
            translate_fn,
            max_batch_size=max_batch_size,
            options=options)
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

def preprocess_example(func, index, raw_example, config=None):
    """Applies preprocessing function on example."""
    if not isinstance(raw_example, dict):
        raise ValueError('example %d is not a JSON object' % index)
    source_text = raw_example.get('text')
    if source_text is None:
        raise ValueError('missing text field in example %d' % index)
    mode = raw_example.get('mode', 'default')
    config = finalize_config(
        config,
        override=raw_example.get('config'),
        options=raw_example.get('options'))

    target_prefix = raw_example.get('target_prefix')
    target_fuzzy = raw_example.get('fuzzy')
    if target_prefix is not None and target_fuzzy is not None:
        raise ValueError("Using both a target prefix and a fuzzy target is currently unsupported")
    if target_prefix is not None:
        target_text = target_prefix
        target_type = "prefix"
    elif target_fuzzy is not None:
        target_text = target_fuzzy
        target_type = "fuzzy"
    else:
        target_text = None
        target_type = None
    if target_type is not None:
        config = config.copy() if config is not None else {}
        config["target_type"] = target_type

    result = func(source_text, target_text, config)

    source_tokens = result[0]
    target_tokens = result[1]
    # Preprocessing may return additional metadata alongside tokens.
    metadata = result[2] if len(result) >= 3 else None

    # Move to the general multiparts representation.
    if not source_tokens or not isinstance(source_tokens[0], list):
        source_tokens = [source_tokens]
        target_tokens = [target_tokens]
        metadata = [metadata]

    return TranslationExample(
        index=index,
        config=config,
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        mode=mode,
        metadata=metadata)

def preprocess_examples(raw_examples, func, config=None):
    """Applies preprocessing on a list of example structures."""
    examples = []
    for i, raw_example in enumerate(raw_examples):
        example = preprocess_example(func, i, raw_example, config=config)
        examples.append(example)
    return examples

def postprocess_output(output, example, func):
    """Applies postprocessing function on a translation output."""
    if example.num_parts > 1:
        # For multi parts inputs, send all parts to the postprocessing.
        tgt_tokens = output.output
        src_context = (example.source_tokens, example.metadata)
        score = sum(output.score) if all(s is not None for s in output.score) else None
        align = None
    else:
        # Otherwise just take the first element and pass metadata only if defined.
        tgt_tokens = output.output[0]
        src_tokens = example.source_tokens[0]
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

def translate_examples(examples, func, max_batch_size=None, options=None):
    """Translates examples."""
    if options is None:
        options = {}
    hypotheses_per_example = collections.defaultdict(list)
    for batch in batch_iterator(examples, max_batch_size=max_batch_size):
        batch_options = options.copy()
        batch_options['mode'] = batch.mode
        batch_hypotheses = func(batch.source_tokens, batch.target_tokens, batch_options)
        if batch_hypotheses is None:
            raise RuntimeError('translation failed or timed out')

        # Gather hypotheses by example id.
        for index, hypotheses in zip(batch.indices, batch_hypotheses):
            hypotheses_per_example[index].append(hypotheses)

    # Merge multi-part hypotheses.
    outputs = []
    for index, hypotheses in six.iteritems(hypotheses_per_example):
        num_hypotheses = len(hypotheses[0])
        outputs.insert(
            index,
            [merge_translation_outputs(part[h] for part in hypotheses)
             for h in range(num_hypotheses)])

    return outputs

def batch_iterator(examples, max_batch_size=None):
    """Yields batch of tokens not larger than max_batch_size."""
    examples_per_mode = collections.defaultdict(list)
    for example in examples:
        examples_per_mode[example.mode].append(example)
    for mode, examples in six.iteritems(examples_per_mode):
        indices = []
        source_tokens = []
        target_tokens = []
        for example in examples:
            indices.extend(example.index for _ in range(example.num_parts))
            source_tokens.extend(example.source_tokens)
            target_tokens.extend(example.target_tokens)
        batch_size = len(source_tokens)
        if max_batch_size is None or batch_size <= max_batch_size:
            yield TranslationBatch(
                indices=indices,
                source_tokens=source_tokens,
                target_tokens=target_tokens,
                mode=mode)
        else:
            offset = 0
            while offset < batch_size:
                lower_bound = offset
                upper_bound = min(offset + max_batch_size, batch_size)
                yield TranslationBatch(
                    indices=indices[lower_bound:upper_bound],
                    source_tokens=source_tokens[lower_bound:upper_bound],
                    target_tokens=target_tokens[lower_bound:upper_bound],
                    mode=mode)
                offset = upper_bound

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
