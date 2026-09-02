from __future__ import annotations

import time
from collections.abc import Callable


def retry[T](operation: Callable[[], T], attempts: int = 3, delay_sec: float = 0.2) -> T:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    last_error: Exception | None = None
    for index in range(attempts):
        try:
            return operation()
        except Exception as error:
            last_error = error
            if index + 1 < attempts:
                time.sleep(delay_sec * (2**index))
    assert last_error is not None
    raise last_error
