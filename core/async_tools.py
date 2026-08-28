import asyncio
import concurrent.futures
import functools
import inspect
from typing import Any, Awaitable, TypeVar, Coroutine, Callable, ParamSpec


T = TypeVar("T")

def run_sync(coro: Awaitable[T] | Coroutine[Any, Any, T]) -> T:
    """
    Execute *coro* and return its result, even when an event loop is
    already running in this thread.

    • In a plain script / unit-test (no loop running)  → `asyncio.run(...)`.
    • In sync code but with a loop already running    → schedule on that
      loop and block in a helper thread (keeps the loop alive).
    • From inside `async def`                         → raise; caller
      should `await` instead.
    """
    # 1) No need to do anything if it isn't awaitable
    if not inspect.isawaitable(coro):
        # The caller gave us a plain value; just return it.
        return coro                                    # type: ignore[return-value]

    # 2) Is there a loop in *this* thread?
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:                                   # → plain sync context
        return asyncio.run(coro)

    # 3) A loop *is* running in this thread
    in_async_context = asyncio.current_task(loop=loop) is not None

    if in_async_context:
        # ──────────────────────────────────────────────────────────────
        # We’re *inside* `async def`.  Block the current loop thread,
        # but execute *coro* in its *own* thread & event loop so it can
        # still make progress.
        # ──────────────────────────────────────────────────────────────
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(lambda: asyncio.run(coro))
            return fut.result()

    # # 3) A loop *is* running in this thread
    # if asyncio.current_task(loop=loop) is not None:    # ← inside async def
    #     raise RuntimeError(
    #         "`run_sync()` called from an async context. "
    #         "Just `await` the coroutine directly."
    #     )

    # 4) We’re in synchronous code (e.g. Jupyter cell, FastAPI on_startup)
    #    but *share* the thread with a running loop.
    def _runner() -> T:
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        return fut.result()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_runner).result()
    

P = ParamSpec("P")

def syncify(func: Callable[P, Awaitable[T]]) -> Callable[P, T]:
    """
    Turn an *async* function or method into a blocking sibling
    that delegates through `run_sync`.

    Usage:

        class Client:
            async def call(self, x): ...
            sync_call = syncify(call)

    or

        @syncify
        async def fetch(url): ...
    """
    @functools.wraps(func)
    def wrapper(*args: P.args, **kw: P.kwargs) -> T:       # type: ignore[valid-type]
        return run_sync(func(*args, **kw))
    return wrapper
