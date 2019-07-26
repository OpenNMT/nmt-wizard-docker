import json
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
                body = json.loads(post_body)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Incoming request: %s", json.dumps(body, ensure_ascii=False))
            except ValueError:
                self.send_error(400, 'badly formatted JSON data')
                return
            self.handle_request(body)

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

        def send_result(self, result):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            data = json.dumps(result)
            if not isinstance(data, six.binary_type):
                data = data.encode("utf-8")
            self.wfile.write(data)

        def handle_request(self, request):
            if 'src' not in request:
                self.send_error(400, 'missing src field')
                return
            results = {'tgt': []}
            if not request['src']:
                self.send_result(results)
                return
            if not isinstance(request['src'], list):
                self.send_error(400, 'src field must be a list')
                return
            timeout = global_timeout
            max_batch_size = global_max_batch_size
            batch_config = config
            request_options = request.get('options')
            if request_options is not None and isinstance(request_options, dict):
                timeout = request_options.get('timeout', timeout)
                max_batch_size = request_options.get('max_batch_size', max_batch_size)
                if 'config' in request_options:
                    batch_config = config_util.merge_config(
                        copy.deepcopy(config), request['options']['config'])
            extra_config = []
            batch_metadata = []
            batch_offsets = []
            batch_tokens = []
            offset = 0
            for src in request['src']:
                local_config = batch_config
                if 'config' in src or 'options' in src:
                    local_config = copy.deepcopy(local_config)
                    if 'config' in src:
                        local_config = config_util.merge_config(local_config, src['config'])
                    if 'options' in src:
                        try:
                            config_util.update_config_with_options(local_config, src['options'])
                        except ValueError as e:
                            self.send_error(400, e.message)
                            return
                data = preprocess_fn(serving_state, src['text'], local_config)
                # Preprocessing may return additional metadata.
                if isinstance(data, tuple):
                    tokens, metadata = data
                else:
                    tokens, metadata = data, None
                # Preprocessing may split input text into multiple parts.
                if tokens and isinstance(tokens[0], list):
                    size = len(tokens)
                    # Flatten the parts in the batch collection.
                    batch_tokens.extend(tokens)
                    batch_metadata.extend(metadata)
                else:
                    size = 1
                    batch_tokens.append(tokens)
                    batch_metadata.append(metadata)
                extra_config.append(local_config)
                batch_offsets.append((offset, offset + size))
                offset += size
            if max_batch_size is not None and len(batch_tokens) > max_batch_size:
                offset = 0
                batch_hypotheses = []
                while offset < len(batch_tokens):
                    lower_bound = offset
                    upper_bound = min(offset + max_batch_size, len(batch_tokens))
                    batch_hypotheses.extend(translate_fn(
                        batch_tokens[lower_bound:upper_bound],
                        backend_info,
                        timeout=timeout))
                    offset = upper_bound
            else:
                batch_hypotheses = translate_fn(
                    batch_tokens, backend_info, timeout=timeout)
            if batch_hypotheses is None:
                self.send_error(504, 'translation request timed out')
                return
            for local_config, offset in zip(extra_config, batch_offsets):
                hypotheses = batch_hypotheses[offset[0]:offset[1]]
                num_parts = offset[1] - offset[0]
                num_hypotheses = len(hypotheses[0])
                src_tokens = batch_tokens[offset[0]:offset[1]]
                src_metadata = batch_metadata[offset[0]:offset[1]]
                result = []
                for h in range(num_hypotheses):
                    if num_parts == 1:
                        src = src_tokens[0]
                        if src_metadata[0] is not None:
                            src = (src, src_metadata[0])
                        tgt = hypotheses[0][h].output
                        scores = hypotheses[0][h].score
                        attention = hypotheses[0][h].attention
                    else:
                        # For multi parts inputs, send all result parts to the postprocessing.
                        src = (src_tokens, src_metadata)
                        tgt = []
                        scores = []
                        attention = None
                        for j in range(num_parts):
                            tgt.append(hypotheses[j][h].output)
                            scores.append(hypotheses[j][h].score)
                    result.append(_build_result(
                        lambda src, tgt: postprocess_fn(serving_state, src, tgt, local_config),
                        src,
                        tgt,
                        scores=scores,
                        attention=attention,
                        num_parts=num_parts))
                results['tgt'].append(result)
            self.send_result(results)

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


def _build_result(postprocess_fn, src_tokens, tgt_tokens, scores=None, attention=None, num_parts=1):
    result = {}
    result['text'] = postprocess_fn(src_tokens, tgt_tokens)
    if scores is not None:
        if num_parts > 1:
            result['score'] = sum(scores)
        else:
            result['score'] = scores
    if attention is not None and num_parts == 1:
        result['align'] = _align(src_tokens, tgt_tokens, attention)
    return result

def _align(src_tokens, tgt_tokens, attention):
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

def _process_is_running(process):
    return process is not None and process.poll() is None
