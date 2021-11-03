"""Define the beat service to interact with the task launcher."""

import contextlib
import time
import threading
import requests

from nmtwizard.logger import get_logger

logger = get_logger(__name__)

_stop_beat = threading.Event()
_beat_thread = None
_last_activity = None
_last_activity_lock = threading.Lock()


def start_beat_service(
    container_id, url, task_id, interval=30, inactivity_timeout=None
):
    """Start a background service that sends HTTP PUT requests to:

    url/task/beat/task_id?container_id=container_id&duration=interval

    every `interval` seconds.
    """
    # If no URL is set, consider the beat service as disabled.
    if url is None or task_id is None:
        logger.warning(
            "CALLBACK_URL or task_id is unset; beat service will be disabled"
        )
        return
    if beat_service_is_running():
        logger.warning("The beat service is already running")
        return

    request_params = {"container_id": container_id, "duration": str(interval * 2)}

    def _beat():
        requests.put("%s/task/beat/%s" % (url, task_id), params=request_params)

    def _beat_loop():
        while True:
            if _stop_beat.wait(interval):
                break
            if inactivity_timeout is not None:
                with _last_activity_lock:
                    if (
                        _last_activity is not None
                        and time.time() - _last_activity > inactivity_timeout
                    ):
                        logger.warning(
                            "No process activity after %d seconds. Stopping the beat requests.",
                            inactivity_timeout,
                        )
                        break
            _beat()

    _beat()  # First beat in the main thread to fail for wrong url.
    logger.info("Starting the beat service to %s with interval %d", url, interval)
    global _beat_thread
    _beat_thread = threading.Thread(target=_beat_loop)
    _beat_thread.daemon = True
    _beat_thread.start()


def stop_beat_service():
    """Stop the beat service."""
    if beat_service_is_running():
        _stop_beat.set()
        _beat_thread.join()
        _stop_beat.clear()


def beat_service_is_running():
    """Returns True if the beat service is currently running."""
    return _beat_thread is not None and _beat_thread.is_alive()


@contextlib.contextmanager
def monitor_activity():
    monitor = _ActivityMonitor()
    monitor.notify()
    yield monitor
    monitor.stop()


class _ActivityMonitor:
    def notify(self):
        global _last_activity
        with _last_activity_lock:
            _last_activity = time.time()

    def stop(self):
        global _last_activity
        with _last_activity_lock:
            _last_activity = None
