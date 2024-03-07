import pathlib
import time
from concurrent.futures.thread import ThreadPoolExecutor
from threading import Event
from typing import Any, Optional

from python_mpv_jsonipc import MPV, MPVError

from .audio import generate_chunks
from .config import Config, get_config
from .subtitle import SRTFile, to_time_format
from .transcribe import whisper_chunk


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
            cancel=self.event,
            path=path,
            position=position,
            language=None,
        )

    def cancel(self, event_data: Any = None):
        self.event.set()

    ## mainloop

    def handle_path(
        self,
        cancel: Event,
        path: str,
        position: float,
        language: Optional[str] = None,
    ):
        config = get_config()

        subtitle = SRTFile(config.subtitle.get_subtitle(path))
        subtitle.clear()
        self.command("sub-add", subtitle.path)

        print("whispering", path)
        self.command("show-text", f"whispering {path}")

        if not language and config.task.language:
            language = config.task.language

        for chunk, start in generate_chunks(path, position, config.task.chunk_duration):
            if cancel.is_set():
                break

            print(f"working: {to_time_format(start)}")
            begin = time.time()
            segments, language = whisper_chunk(chunk, language)
            print("got segments")
            with subtitle.open("a"):
                for segment in segments:
                    print(segment)
                    subtitle.write(
                        start + segment.start,
                        start + segment.end,
                        segment.text,
                    )
            duration = time.time() - begin
            print(f"progress: {to_time_format(start)} ({duration:.2f}s)")
            self.command("sub-reload")
        else:
            print("completed whisper for", path)
            self.command("show-text", f"completed whisper for {path}")

        print("ending whisper for", path)
