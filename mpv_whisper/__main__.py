import sys

from .monitor import MPVMonitor
from .transcribe import get_model

get_model()
monitor = MPVMonitor()
if len(sys.argv) > 1:
    monitor.command("loadfile", sys.argv[1])

monitor.block()
