import logging
import time
from threading import Event
from typing import Optional

from .audio import generate_chunks
from .config import get_config
from .subtitle import SRTFile, to_time_format
from .transcribe import whisper_chunk


def core_loop(
    path: str,
    position: float,
    language: Optional[str] = None,
    cancel: Optional[Event] = None,
):
    cancel = cancel or Event()

    log = logging.getLogger("core_loop")

    config = get_config()

    subtitle = SRTFile(config.subtitle.get_subtitle(path))
    subtitle.clear()
    yield subtitle

    log.info("whispering %s", path)
    language = language or config.transcribe.language

    for chunk, start in generate_chunks(path, position, config.transcribe.chunk_duration):
        if cancel.is_set():
            break
        log.debug("working: %s", to_time_format(start))
        begin = time.time()
        segments, language = whisper_chunk(chunk, language)
        log.debug("got segments")
        with subtitle.open("a"):
            for segment in segments:
                subtitle.write(
                    start + segment.start,
                    start + segment.end,
                    segment.text,
                )
                yield segment
        duration = time.time() - begin
        log.info("progress: %s (%ss)", to_time_format(start), f"{duration:.2f}")
    else:
        log.info("completed whisper for %s", path)
        yield True

    log.info("ending whisper for %s", path)
