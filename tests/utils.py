import asyncio
import builtins
import contextlib
import queue
import threading

from aiohttp import web


# Python's built in http.server does not support the Range header.  aoihttp does
def _run_aiohttp_server(app, loop, ready_event, stop_event, msg_queue):
    asyncio.set_event_loop(loop)
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "127.0.0.1", 8080)
    loop.run_until_complete(site.start())

    ready_event.set()
    msg_queue.put(site.name)
    # Wait for the stop event to be set
    loop.run_until_complete(stop_event.wait())

    # Cleanup when stop event is set
    loop.run_until_complete(runner.cleanup())
    loop.close()


@contextlib.contextmanager
def static_http_server(static_dir):
    ready_event = threading.Event()
    stop_event = asyncio.Event()
    msg_queue = queue.Queue()

    app = web.Application()
    app.add_routes([web.static("/", static_dir)])

    loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=_run_aiohttp_server,
        args=(app, loop, ready_event, stop_event, msg_queue),
        daemon=True,
    )
    thread.start()

    if ready_event.wait(timeout=10):
        url = msg_queue.get()
        yield url

    loop.call_soon_threadsafe(stop_event.set)
    thread.join()


def simple_open_read(filename, *args, **kwargs):
    """Open a file, returning a file-like object with some methods supported"""

    class _SimpleFile:
        def __init__(self, filename):
            self._file = builtins.open(filename, "rb")

            self.read = self._file.read
            self.readline = self._file.readline
            self.seek = self._file.seek
            self.tell = self._file.tell
            self.close = self._file.close

        def __enter__(self, *args, **kwargs):
            return self

        def __exit__(self, *args, **kwargs):
            return self.close()

    return _SimpleFile(filename)
