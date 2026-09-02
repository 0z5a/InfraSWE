from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class AsyncJobQueue:
    def __init__(self, concurrency: int = 1) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        self._semaphore = asyncio.Semaphore(concurrency)

    async def submit(self, job: Callable[[], Awaitable[T]]) -> T:
        async with self._semaphore:
            return await job()
