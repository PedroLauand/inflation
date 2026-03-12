from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from time import perf_counter
from typing import Callable, Iterator

from tqdm.auto import tqdm as AutoTqdm
from tqdm.std import tqdm as StdTqdm

__all__ = ["make_tqdm", "progress_stage"]


def _stream_is_tty(stream) -> bool:
    isatty = getattr(stream, "isatty", None)
    if isatty is None:
        return False
    try:
        return bool(isatty())
    except OSError:
        return False


def _should_use_line_progress(stream) -> bool:
    return bool(os.environ.get("SLURM_JOB_ID")) or not _stream_is_tty(stream)


def _write_progress_line(stream, text: str) -> None:
    if not text:
        return
    stream.write(str(text))
    stream.write("\n")
    flush = getattr(stream, "flush", None)
    if callable(flush):
        flush()


def _normalize_stage_label(message: str) -> str:
    if message.endswith("..."):
        return message[:-3]
    if message.endswith("."):
        return message[:-1]
    return message


class _LineTqdm(StdTqdm):
    @staticmethod
    def status_printer(file):
        fp = file

        def print_status(s):
            _write_progress_line(fp, str(s))

        return print_status


def make_tqdm(*args, **kwargs):
    stream = kwargs.get("file")
    if stream is None:
        stream = sys.stdout
        kwargs["file"] = stream

    if _should_use_line_progress(stream):
        kwargs["position"] = 0
        return _LineTqdm(*args, **kwargs)
    return AutoTqdm(*args, **kwargs)


@contextmanager
def progress_stage(
    start_message: str,
    *,
    enabled: bool = True,
    file=None,
    end_message: str | Callable[[float], str] | None = None,
) -> Iterator[None]:
    if not enabled:
        yield
        return

    stream = sys.stdout if file is None else file
    _write_progress_line(stream, start_message)
    start = perf_counter()
    try:
        yield
    except Exception:
        elapsed = perf_counter() - start
        _write_progress_line(
            stream,
            f"{_normalize_stage_label(start_message)} failed after {elapsed:.2f}s",
        )
        raise

    elapsed = perf_counter() - start
    if end_message is None:
        final_message = f"{_normalize_stage_label(start_message)} complete in {elapsed:.2f}s"
    elif callable(end_message):
        final_message = end_message(elapsed)
    else:
        final_message = end_message.format(elapsed=elapsed)
    _write_progress_line(stream, final_message)
