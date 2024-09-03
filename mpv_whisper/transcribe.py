import itertools
import logging
from functools import cache
from typing import Any, Iterable, Literal, Optional

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

from .config import get_config

log = logging.getLogger("transcribe")


@cache
def get_model():
    config = get_config()
    model = WhisperModel(config.model.model, **config.model.args)
    log.info("loaded whisper [%s] with %s", config.model.model, config.model.args)
    return model


def _do_chunk(
    task: Literal["transcribe", "translate"],
    audio: Any,
    lang: Optional[tuple[str, float]] = None,
):
    segments, info = get_model().transcribe(
        audio,
        task=task,
        language=lang[0] if lang else None,
        **get_config().model.task_args,
    )

    return segments, lang or (info.language, info.language_probability)


def whisper_chunk(
    audio: Any,
    language: Optional[str] = None,
    task: Literal[
        "transcribe", "translate", "both"
    ] = get_config().transcribe.foreign_lang_behaviour,
):
    segments: list[Iterable[Segment]] = []
    lang = (language, 1.0) if language else None

    if task in ("transcribe", "both"):
        log.debug("transcribing")
        segment, lang = _do_chunk("transcribe", audio, lang)
        segments.append(segment)

    if task in ("translate", "both"):
        log.debug("translating")
        segment, lang = _do_chunk("translate", audio, lang)
        segments.append(segment)

    assert lang
    if not language and lang[1] >= get_config().transcribe.confidence_threshold:
        log.info("Detected language '%s' with probability %s", lang[0], f"{lang[1]:.5f}")
        language = lang[0]

    assert segments
    return itertools.chain(*segments), language
