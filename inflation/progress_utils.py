from __future__ import annotations

import os
import sys

from tqdm.auto import tqdm as AutoTqdm
from tqdm.std import tqdm as StdTqdm

__all__ = ["make_tqdm"]


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


class _LineTqdm(StdTqdm):
    @staticmethod
    def status_printer(file):
        fp = file
        fp_flush = getattr(fp, "flush", lambda: None)

        def print_status(s):
            text = str(s)
            if not text:
                return
            fp.write(text)
            fp.write("\n")
            fp_flush()

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
