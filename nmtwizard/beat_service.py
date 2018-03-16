"""Define the beat service to interact with the task launcher."""

import time
import threading
import requests

from nmtwizard.logger import get_logger

logger = get_logger(__name__)


def start_beat_service(container_id, url, task_id, interval=30):
    """Start a background service that sends a HTTP GET request to:

    url/beat/task_id?container_id=container_id&duration=interval

    every `interval` seconds.
    """
    # If no URL is set, consider the beat service as disabled.
    if url is None or task_id is None:
        logger.warning('CALLBACK_URL or task_id is unset; beat service will be disabled')
        return

    request_params = {
        'container_id': container_id,
        'duration': str(interval * 2)
    }

    def _beat():
        requests.get('%s/task/beat/%s' % (url, task_id), params=request_params)

    def _beat_loop():
        while True:
            time.sleep(interval)
            _beat()

    _beat()  # First beat in the main thread to fail for wrong url.
    logger.info('Starting the beat service to %s with interval %d', url, interval)
    notify_thread = threading.Thread(target=_beat_loop)
    notify_thread.daemon = True
    notify_thread.start()
