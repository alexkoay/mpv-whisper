import os.path
import pathlib
from functools import cache
from typing import Any, Literal, Optional

import cattrs
from attr import dataclass, field

try:
    import tomllib as toml
except ImportError:
    import toml


def expand_path(path: str):
    path = os.path.expandvars(path)
    path = os.path.expanduser(path)
    return pathlib.Path(path)


def ensure_path(path: str):
    p = expand_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(kw_only=True)
class ModelConfig:
    model: str = "base"
    args: dict[str, Any] = field(factory=dict)
    task_args: dict[str, Any] = field(factory=lambda: {"beam_size": 5})


@dataclass(kw_only=True)
class TranscribeConfig:
    language: Optional[str] = None
    foreign_lang_behaviour: Literal["transcribe", "translate", "both"] = "transcribe"
    confidence_threshold: float = 0.95

    chunk_duration: float = 15.0


@dataclass(kw_only=True)
class MpvConfig:
    executable: Optional[str] = None
    start_mpv: bool = True
    start_args: dict[str, Any] = field(factory=dict)
    ipc_socket: Optional[str] = None

    toggle_binding: str = "ctrl+."


@dataclass(kw_only=True)
class SubtitleConfig:
    path: pathlib.Path = field(
        default="~/.config/mpv-whisper/subs",
        converter=ensure_path,
    )
    only_network: bool = False

    def get_subtitle(self, fname: str):
        if not self.only_network or "://" in fname:
            return (self.path / pathlib.Path(fname).stem).with_suffix(".whisper.srt")
        else:
            return pathlib.Path(fname).with_suffix(".whisper.srt")


@dataclass(kw_only=True)
class Config:
    model: ModelConfig = field(factory=ModelConfig)
    transcribe: TranscribeConfig = field(factory=TranscribeConfig)
    mpv: MpvConfig = field(factory=MpvConfig)
    subtitle: SubtitleConfig = field(factory=SubtitleConfig)


def load_config_paths(*paths: pathlib.Path):
    for p in paths:
        if not p.exists():
            continue

        print(f"using configuration from {p}")
        raw: Any = toml.load(p)
        conv = cattrs.GenConverter(forbid_extra_keys=True)
        return conv.structure(raw, Config)

    raise RuntimeError("could not find configuration file")


@cache
def get_config() -> Config:
    return load_config_paths(
        expand_path(".") / "mpv-whisper.toml",
        expand_path("~/.config/mpv-whisper") / "config.toml",
        expand_path(__file__).parent / "config.toml",
    )
