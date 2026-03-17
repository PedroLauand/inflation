import os
import unittest


def _slow_tests_enabled() -> bool:
    flag = os.environ.get("RUN_SLOW_TESTS", "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def require_slow_tests(module_name: str) -> None:
    if not _slow_tests_enabled():
        raise unittest.SkipTest(
            f"{module_name} is part of the slow integration suite. "
            "Set RUN_SLOW_TESTS=1 to include it."
        )


def slow_test(obj):
    return unittest.skipUnless(
        _slow_tests_enabled(),
        "Slow integration test. Set RUN_SLOW_TESTS=1 to include it.",
    )(obj)
