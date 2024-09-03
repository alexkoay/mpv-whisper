import logging
from typing import Optional

import click

from .coreloop import core_loop
from .monitor import MPVMonitor
from .subtitle import SRTFile
from .transcribe import get_model


@click.command
@click.argument("path", required=False, default=None)
@click.option("--loglevel", default="INFO")
def cli(path: Optional[str], loglevel: str):
    logging.basicConfig(level=loglevel)
    logging.getLogger("faster_whisper").disabled = True
    get_model()
    monitor = MPVMonitor()
    if path:
        monitor.command("loadfile", path)

    monitor.block()


@click.command
@click.argument("path", required=True)
@click.option("--position", default=0.0)
@click.option("--language", default=None)
@click.option("--loglevel", default="INFO")
@click.option("--echo/--no-echo", default=False)
def direct(
    path: str,
    position: float,
    language: Optional[str],
    loglevel: str,
    echo: bool,
):
    logging.basicConfig(level=loglevel)
    logging.getLogger("faster_whisper").disabled = True
    get_model()
    for segment in core_loop(path=path, position=position, language=language):
        if segment is True:
            continue
        elif isinstance(segment, SRTFile):
            logging.info("writing to %s", segment.path)
        elif echo:
            print(segment.text)
