import json
import signal
import threading
import collections
import socket
import time
import six

from six.moves import SimpleHTTPServer
from six.moves import socketserver

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
      preprocess_fn: A callable taking (serving_state, text) and returning tokens.
      translation_fn: A callable that forwards the request to the translation backend.
      postprocess_fn: A callable taking (serving_state, tokens) and returning text.
    """
    global backend_process
    global backend_info
    backend_process, backend_info = backend_service_fn()
    global_timeout = None
    global_max_batch_size = None
    if config is not None and isinstance(config, dict):
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
            if not _process_is_running(backend_process):
                self.send_error(503, 'backend service is unavailable')
                return
            content_len = int(self.headers.getheader('content-length', 0))
            if content_len == 0:
                self.send_error(400, 'missing request data')
                return
            post_body = self.rfile.read(content_len)
            try:
                body = json.loads(post_body)
            except ValueError:
                self.send_error(400, 'badly formatted JSON data')
                return
            self.handle_request(body)

        def unload_model(self):
            global backend_process
            if _process_is_running(backend_process):
                backend_process.terminate()
            backend_process = None
            self.send_response(200)

        def reload_model(self):
            global backend_process
            global backend_info
            if _process_is_running(backend_process):
                backend_process.terminate()
            backend_process, backend_info = backend_service_fn()
            self.send_response(200)

        def send_result(self, result):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result))

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
            if 'options' in request and isinstance(request['options'], dict):
                timeout = request['options'].get('timeout', timeout)
                max_batch_size = request['options'].get('max_batch_size', max_batch_size)
            batch_tokens = [
                preprocess_fn(serving_state, src['text']) for src in request['src']]
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
            for src_tokens, hypotheses in zip(batch_tokens, batch_hypotheses):
                result = []
                for output in hypotheses:
                    tgt = {}
                    tgt['text'] = postprocess_fn(serving_state, output.output)
                    if output.score is not None:
                        tgt['score'] = output.score
                    if output.attention is not None:
                        tgt['align'] = _align(src_tokens, output.output, output.attention)
                    result.append(tgt)
                results['tgt'].append(result)
            self.send_result(results)

    try:
        frontend_server = socketserver.TCPServer((host, port), ServerHandler)
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
    return process is not None and process.poll() == None
