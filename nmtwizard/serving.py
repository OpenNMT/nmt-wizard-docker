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
                                   "options",
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
                 backend_service_fn,
                 translate_fn,
                 preprocessor=None,
                 postprocessor=None,
                 backend_info_fn=None,
                 rebatch_request=True):
    """Start a serving service.

    This function will only return on SIGINT or SIGTERM signals.

    Args:
      host: The hostname of the service.
      port: The port used by the service.
      backend_service_fn: A callable to start the framework dependent backend service.
      translation_fn: A callable that forwards the request to the translation backend.
      backend_info_fn: A callable returning some information about the backend service,
        and whether it can accept new requests or not.
      preprocessor: A Processor instance for preprocessing.
      postprocessor: A Processor instance for postprocessing.
      rebatch_request: If True, incoming requests are rebatched according to
        max_batch_size. Otherwise, max_batch_size is passed as a translation option
        to translate_fn which takes responsibility over batching.
    """
    global backend_process
    global backend_info
    backend_process, backend_info = backend_service_fn()
    serving_config = config.get('serving')
    if serving_config is None:
        serving_config = {}
    global_timeout = serving_config.get('timeout')
    global_max_batch_size = serving_config.get('max_batch_size')

    def _backend_is_reachable():
        return (backend_info is not None
                and backend_process is None or _process_is_running(backend_process))

    class ServerHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
        def _send_response(self, data, status=200):
            self.send_response(status)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(six.ensure_binary(json.dumps(data)))

        def _send_error(self, status, message):
            data = {"message": message}
            self._send_response(data, status=status)

        def do_GET(self):
            if self.path == '/status':
                self.status()
            elif self.path == '/health':
                self.health()
            else:
                self._send_error(404, 'invalid route %s' % self.path)

        def do_POST(self):
            if self.path == '/translate':
                self.translate()
            elif self.path == '/unload_model':
                self.unload_model()
            elif self.path == '/reload_model':
                self.reload_model()
            else:
                self._send_error(404, 'invalid route %s' % self.path)

        def translate(self):
            if not _backend_is_reachable():
                self._send_error(503, 'backend service is unavailable')
                return
            header_fn = (
                self.headers.getheader if hasattr(self.headers, "getheader")
                else self.headers.get)
            content_len = int(header_fn('content-length', 0))
            if content_len == 0:
                self._send_error(400, 'missing request data')
                return
            post_body = self.rfile.read(content_len)
            try:
                result = run_request(
                    json.loads(six.ensure_str(post_body)),
                    functools.partial(translate_fn, backend_info),
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    config=config,
                    rebatch_request=rebatch_request,
                    max_batch_size=global_max_batch_size,
                    timeout=global_timeout)
            except ValueError as e:
                self._send_error(400, str(e))
            except RuntimeError as e:
                self._send_error(504, str(e))
            else:
                self._send_response(result)

        def health(self):
            if not _backend_is_reachable():
                self._send_error(503, 'backend service is unavailable')
                return
            if backend_info_fn is None:
                self._send_response({})
                return
            info, available = backend_info_fn(serving_config, backend_info)
            self._send_response(info, status=200 if available else 503)

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
                translate_fn,
                preprocessor=None,
                postprocessor=None,
                config=None,
                rebatch_request=True,
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
        max_batch_size = options.get('max_batch_size', max_batch_size)
        if not rebatch_request and max_batch_size is not None:
            options['max_batch_size'] = max_batch_size
            max_batch_size = None

        examples = preprocess_examples(
            src,
            preprocessor,
            config=config,
            config_override=options.get('config'))
        outputs = translate_examples(
            examples,
            translate_fn,
            max_batch_size=max_batch_size,
            options=options)
        results = postprocess_outputs(outputs, examples, postprocessor)

    return {'tgt': results}

def finalize_config(config, override=None, options=None):
    """Finalizes the configuration with possible override and options."""
    if config is not None:
        if config_util.is_v2_config(config):
            if override:
                raise ValueError("Configuration override is not supported for V2 "
                                 "configurations")
            if options:
                options = config_util.read_options(config, options)
            config = None
        else:
            if override or options:
                config = copy.deepcopy(config)
                if override:
                    config_util.merge_config(config, override)
                if options:
                    config_util.read_options(config, options)
                    options = None
    return config, options

def preprocess_example(preprocessor, index, raw_example, config=None, config_override=None):
    """Applies preprocessing function on example."""
    if not isinstance(raw_example, dict):
        raise ValueError('example %d is not a JSON object' % index)
    source_text = raw_example.get('text')
    if source_text is None:
        raise ValueError('missing text field in example %d' % index)
    mode = raw_example.get('mode', 'default')

    example_config_override = raw_example.get('config')
    if example_config_override:
        if config_override:
            config_override = config_util.merge_config(
                copy.deepcopy(config_override), example_config_override)
        else:
            config_override = example_config_override
    config, options = finalize_config(
        config,
        override=config_override,
        options=raw_example.get('options'))

    target_prefix = raw_example.get('target_prefix')
    target_fuzzy = raw_example.get('fuzzy')
    if target_prefix is not None and target_fuzzy is not None:
        raise ValueError("Using both a target prefix and a fuzzy target is currently unsupported")

    target_text = None
    target_name = None
    if target_prefix is not None:
        target_text = target_prefix
    elif target_fuzzy is not None:
        target_text = target_fuzzy
        target_name = "fuzzy"

    if preprocessor is None:
        source_tokens = source_text
        target_tokens = None
        metadata = None
    else:
        source_tokens, target_tokens, metadata = preprocessor.process_input(
            source_text,
            target=target_text,
            target_name=target_name,
            config=config,
            options=options,
        )

    # Move to the general multiparts representation.
    if not source_tokens or not isinstance(source_tokens[0], list):
        source_tokens = [source_tokens]
        target_tokens = [target_tokens]
        metadata = [metadata]

    return TranslationExample(
        index=index,
        config=config,
        options=options,
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        mode=mode,
        metadata=metadata)

def preprocess_examples(raw_examples, preprocessor, config=None, config_override=None):
    """Applies preprocessing on a list of example structures."""
    examples = []
    for i, raw_example in enumerate(raw_examples):
        example = preprocess_example(
            preprocessor, i, raw_example, config=config, config_override=config_override)
        examples.append(example)
    return examples

def postprocess_output(output, example, postprocessor):
    """Applies postprocessing function on a translation output."""

    # Send all parts to the postprocessing.
    if postprocessor is None:
        text = output.output[0]
        score = None
        align = None
    else:
        tgt_tokens = output.output
        src_tokens = example.source_tokens
        text = postprocessor.process_input(
            src_tokens,
            tgt_tokens,
            metadata=example.metadata,
            config=example.config,
            options=example.options,
        )
        score = sum(output.score) if all(s is not None for s in output.score) else None
        attention = output.attention
        if attention and len(attention) == 1:
            attention = attention[0]
            align = align_tokens(src_tokens, tgt_tokens, attention) if attention else None
        else:
            align = None

    result = {'text': text}
    if score is not None:
        result['score'] = score
    if align is not None:
        result['align'] = align
    return result

def postprocess_outputs(outputs, examples, postprocessor):
    """Applies postprocess on model outputs."""
    results = []
    for hypotheses, example in zip(outputs, examples):
        results.append([
            postprocess_output(hypothesis, example, postprocessor)
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
