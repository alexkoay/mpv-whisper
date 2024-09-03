import pathlib
from concurrent.futures.thread import ThreadPoolExecutor
from threading import Event
from typing import Any, Optional

from python_mpv_jsonipc import MPV, MPVError

from .config import Config, get_config
from .coreloop import core_loop
from .subtitle import SRTFile


def simple_unstructure(arg: Any) -> Any:
    if isinstance(arg, pathlib.Path):
        return str(arg)
    return arg


class MPVMonitor:
    def __init__(self, config: Optional[Config] = None):
        if not config:
            config = get_config()

        self.shutdown = Event()
        self.mpv = MPV(
            mpv_location=config.mpv.executable,
            start_mpv=config.mpv.start_mpv,
            ipc_socket=config.mpv.ipc_socket,
            quit_callback=self.shutdown.set,
            **config.mpv.start_args,
        )
        self.enabled = True

        self.pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="whisper",
        )
        self.event = Event()

        self.register()
        self.start()

    ## interface

    def command(self, name: str, *args: Any) -> Any:
        """Send a command to the MPV instance via the JSON IPC interface"""
        return self.mpv.command(name, *map(simple_unstructure, args))  # type: ignore

    def register(self):
        """Register the required keybindings to control mpv-whisper"""
        self.mpv.bind_key_press(get_config().mpv.toggle_binding, self.toggle)
        self.mpv.bind_event("start-file", self.start)
        self.mpv.bind_event("end-file", self.cancel)

    def block(self):
        """Wait for MPV to shutdown"""
        self.shutdown.wait()

    ## events

    def toggle(self):
        self.enabled = not self.enabled
        if self.enabled:
            self.command("show-text", "enabling mpv-whisper")
            self.start()
        else:
            self.command("show-text", "disabling mpv-whisper")
            self.cancel()

    def start(self, event_data: Any = None):
        self.cancel()  # ensure that any existing thread is cancelled
        if not self.enabled:
            return

        path: str = self.command("get_property", "path")
        if not path:
            return

        try:
            position = self.command("get_property", "position")
        except MPVError:
            position = 0

        self.event = Event()
        self.pool.submit(
            self.handle_path,
            path=path,
            position=position,
            cancel=self.event,
            language=None,
        )

    def cancel(self, event_data: Any = None):
        self.event.set()

    ## mainloop

    def handle_path(
        self,
        *,
        cancel: Event,
        path: str,
        position: float,
        language: Optional[str] = None,
    ):
        for progress in core_loop(
            path=path,
            position=position,
            language=language,
            cancel=cancel,
        ):
            if isinstance(progress, SRTFile):
                self.command("sub-add", progress.path)
            elif progress is True:
                self.command("show-text", f"completed whisper for {path}")
            else:
                self.command("sub-reload")
