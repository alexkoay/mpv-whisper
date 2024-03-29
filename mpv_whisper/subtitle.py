import pathlib
import textwrap
from typing import Any, Literal, Optional, TextIO, Type


def to_time_format(pos: float):
    secs = pos % 60
    mins = int((pos // 60) % 60)
    hours = int(pos // 3600)
    return f"{hours:02d}:{mins:02d}:{secs:06.3f}".replace(".", ",")


class SRTFile:
    def __init__(self, path: pathlib.Path, clear: bool = True):
        self.path = path
        self.handle: Optional[TextIO] = None
        self.count = 0

        if clear:
            self.clear()

    def __enter__(self):
        assert self.handle
        return self

    def __exit__(self, exc_type: Type[Exception], exc_value: Exception, traceback: Any):
        assert self.handle
        self.handle.close()
        self.handle = None

    def open(self, mode: Literal["a", "w"] = "a"):
        assert not self.handle
        self.handle = open(self.path, mode=mode, encoding="utf8")
        return self

    def clear(self):
        assert not self.handle
        with self.open("w") as sub:
            sub.write(0, 0.5, "generated by mpv-whisper")

    def write(self, start: float, end: float, text: str, wrap: Optional[int] = None):
        assert self.handle

        text_str = text.strip()
        if wrap:
            text_str = textwrap.fill(text_str, width=wrap)

        start_str = to_time_format(start)
        end_str = to_time_format(end)
        self.handle.write(f"{self.count}\n{start_str} --> {end_str}\n{text}\n\n")
        self.count += 1
