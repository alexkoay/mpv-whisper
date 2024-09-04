import gc
import itertools
from typing import Any, BinaryIO, Iterable, Iterator, Union

import av
import numpy as np
from more_itertools import peekable


def _ignore_invalid_frames(frames: Iterable[Any]):
    iterator = iter(frames)

    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except av.InvalidDataError:
            continue


def _chunk_frames(
    frames: Iterable[Any],
    start: float,
    duration: float,
) -> Iterable[Any]:
    for t, chunk in itertools.groupby(frames, key=lambda frame: (frame.time - start) // duration):
        if t < 0:
            continue
        yield chunk


def _group_frames(frames: Iterable[Any], num_samples: int | None = None) -> Iterable[Any]:
    fifo = av.AudioFifo()

    for frame in frames:
        frame.pts = None  # Ignore timestamp check.
        fifo.write(frame)

        if num_samples is not None and fifo.samples >= num_samples:
            yield fifo.read()

    if fifo.samples > 0:
        yield fifo.read()


def _resample_frames(frames: Iterable[Any], resampler: av.AudioResampler) -> Iterable[Any]:
    # Add None to flush the resampler.
    for frame in itertools.chain(frames, [None]):
        yield from resampler.resample(frame)


def generate_chunks(
    input_file: Union[str, BinaryIO],
    start: float,
    duration: float,
    *,
    sampling_rate: int = 16000,
) -> Iterator[tuple[Any, float]]:
    with av.open(input_file, mode="r", metadata_errors="ignore") as container:
        container.seek(int(start * 1_000_000))
        frames = container.decode(audio=0)

        frames = peekable(frames)
        first = frames.peek()
        time_base = first.time_base

        frames = _ignore_invalid_frames(frames)
        chunks = _chunk_frames(frames, start, duration)
        for chunk in chunks:
            resampler = av.AudioResampler(
                format="s16",
                layout="mono",
                rate=sampling_rate,
            )

            frames = _group_frames(chunk, 500000)
            frames = _resample_frames(frames, resampler)
            frames = list(frames)

            audio = np.concatenate(
                [frame.to_ndarray().reshape(-1) for frame in frames],
                axis=0,
            )

            yield (
                audio.astype(np.float32) / 32768.0,
                float(frames[0].dts * time_base),
            )

            # It appears that some objects related to the resampler are not freed
            # unless the garbage collector is manually run.
            del resampler
            gc.collect()
