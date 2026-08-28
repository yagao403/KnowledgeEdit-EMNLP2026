from abc import abstractmethod
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, replace, asdict, is_dataclass, fields
import json
import httpx
import logging
import os
from pathlib import Path
import sys
from tenacity import (
    retry,
    wait_exponential_jitter,
    retry_if_exception,
    RetryCallState,
    stop_after_attempt,
)
import time
from typing import TypeVar, Generic, Callable, Literal, overload

from core import is_docker
from core.async_tools import syncify
from core.message import Message, Section, Tag
from core.steps import Messages
from core.stats import Stats
from core.utils import Colors


# Silence noisy loggers
for noisy in ("httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def print_response(content: str | list[dict], color: str = Colors.BLUE):
    """content is either a generated text (str) or the content of the last message (list[dict])."""
    if isinstance(content, list):
        text = ""
        for section in content:
            if isinstance(section, str):
                text += section
            if section["type"] == "text":
                text += section["text"]
            elif section["type"] == "image":
                text += f"[Image: {section['image']}]"
            else:
                text += str(section)
    else:
        text = content
    print(color + text, end="", flush=True, file=sys.__stdout__)
    time.sleep(0.01)  # This is needed to ensure the output is flushed in Jupyter notebooks


def warn_before_sleep(retry_state: RetryCallState) -> None:
    """
    Called by Tenacity after a failed attempt but before the next sleep.
    """
    assert retry_state.outcome is not None
    exc = retry_state.outcome.exception()
    attempt_no = retry_state.attempt_number
    # When Tenacity has decided to give up, next_action is None.
    next_wait = retry_state.next_action.sleep if retry_state.next_action else 0
    print(
        f"Attempt {attempt_no} failed with {exc.__class__.__name__} - retrying in {next_wait: .2f} s",
        file=sys.__stdout__,
    )


def is_retryable_httpx_error(exc):
    return isinstance(exc, httpx.RequestError)  # e.g. network problems, timeouts


class Completion(str):
    truncated: bool

    def __new__(cls, content, truncated=False):
        obj = super(Completion, cls).__new__(cls, content)
        obj.truncated = truncated
        return obj

    def __add__(self, other: str):
        if isinstance(other, str):
            return Completion(super().__add__(other))
        else:
            raise TypeError(f"Unsupported type for addition: {type(other)}")


class ServerError(Exception):
    """Custom exception for server-side errors."""

# slots=True does not work with autoreload. The same functionality is implemented manually.
@dataclass
class BaseSamplingParams:
    def __setattr__(self, name, value):
        if not self.__dict__.get('_initialized', False):
            # During __init__, allow everything
            object.__setattr__(self, name, value)
        else:
            allowed = self._existing_attributes()
            if name not in allowed:
                raise AttributeError(f"Parameter '{name}' is not a sampling parameter")
            object.__setattr__(self, name, value)

    def __post_init__(self):
        # Mark the object as initialized after __init__ is complete
        object.__setattr__(self, '_initialized', True)

    @classmethod
    def _existing_attributes(cls):
        allowed = set()
        for class_ in cls.__mro__:
            if is_dataclass(class_):
                allowed.update(f.name for f in fields(class_))
        return allowed

    max_tokens: int = 5000
    temperature: float = 1.0
    min_p: float = 0.0
    top_p: float = 1.0


SP = TypeVar("SP", bound="BaseSamplingParams")


class BaseClient(Generic[SP]):
    model: str | None = None
    default_sampling_params: SP

    @overload  # n == 1 -> Completion
    async def call(
        self,
        messages: Sequence[Message],
        n: Literal[1] = 1,
        /, *,
        stats: Stats | None = None,
        stop_seq: list[str] | str | None = None,
        verbose: bool = False,
        continue_last_message: bool = False,
        print_last_message: bool = True,
        heartbeat_fn: Callable | None = None,
        save_path: Path | None = None,
        **sampling_params,
    ) -> Completion: ...
    @overload  # n != 1 -> list[Completion]
    async def call(
        self,
        messages: Sequence[Message],
        n: int = 1,  # n != 1
        /, *,
        stats: Stats | None = None,
        stop_seq: list[str] | str | None = None,
        verbose: bool = False,
        continue_last_message: bool = False,
        print_last_message: bool = True,
        heartbeat_fn: Callable | None = None,
        save_path: Path | None = None,
        **sampling_params,
    ) -> list[Completion]: ...

    async def call(
        self,
        messages: Sequence[Message],
        n: int = 1,
        /, *,
        stats: Stats | None = None,
        stop_seq: list[str] | str | None = None,
        verbose: bool = False,
        continue_last_message: bool = False,
        print_last_message: bool = True,
        heartbeat_fn: Callable | None = None,
        save_path: Path | None = None,
        **sampling_params,
    ) -> Completion | list[Completion]:
        if save_path:
            if n > 1:
                raise NotImplementedError("Saving multiple completions is not supported")
            else:
                _save_messages(messages, Completion(""), continue_last_message, save_path)

        completions = await self._call_impl(
            messages,
            n=n,
            stop_seq=stop_seq,
            verbose=verbose,
            continue_last_message=continue_last_message,
            print_last_message=print_last_message,
            heartbeat_fn=heartbeat_fn,
            **sampling_params,
        )

        if stats:
            stats.update()  # FIXME: we aren't updating anything useful

        if n == 1:
            res = completions[0]
            if save_path:
                _save_messages(messages, res, continue_last_message, save_path)
            return res
        return completions

    sync_call = syncify(call)

    @abstractmethod
    async def _call_impl(
        self,
        messages: Sequence[Message],
        *,
        n: int,
        stop_seq: list[str] | str | None,
        verbose: bool,
        continue_last_message: bool,
        print_last_message: bool,
        heartbeat_fn: Callable | None,
        **sampling_params,
    ) -> list[Completion]: ...

    def check_messages_empty(self, messages: Sequence[Message]) -> None:
        for msg in messages:
            if not msg.sections:
                raise ValueError(f"Empty message in history: {msg}")


def _save_messages(
    messages: Sequence[Message],
    completion: Completion,
    continue_last_message: bool,
    path: Path,
):
    tags = {Tag.COMPLETION}
    if completion.truncated:
        tags.add(Tag.TRUNCATED)
    section = Section(completion, tags=tags)
    if continue_last_message:
        msgs = list(messages[:-1]) + [
            messages[-1].copy(
                sections=messages[-1].sections + [section]
            )
        ]
    else:
        msgs = list(messages) + [Message("assistant", [section])]
    Messages(msgs).to_xml_path(path)


@dataclass
class ClientSamplingParams(BaseSamplingParams):
    stop: list[str] | str | None = None
    include_stop_str_in_output: bool = False
    stop_token_ids: list[int] | None = None  # Token IDs at which to stop generation


class Client(BaseClient[ClientSamplingParams]):
    base_url: str
    timeout: int = 120

    def __init__(
        self,
        model: str | None,
        base_url: str | None = None,
    ):
        """
        model can be:
          - short model name, e.g., "qwen3-32b"
          - f"agent-{base_model}/{adapter_name}", e.g., "agent-qwen3-32b/adapter_name"
          - None, which means whatever model the server is serving.
        """
        self.model = model
        if base_url is None:
            default_base_url = "http://localhost:8000/" if not is_docker() else "http://host.docker.internal:8000/"
            self.base_url = os.environ.get("VLLM_BASE_URL", default_base_url)
        else:
            self.base_url = base_url

        self.default_sampling_params = ClientSamplingParams()

    async def _call_impl(
        self,
        messages: Sequence[Message],
        *,
        n: int,
        stop_seq: list[str] | str | None,
        verbose: bool,
        continue_last_message: bool,
        print_last_message: bool,
        heartbeat_fn: Callable | None,
        **sampling_params,
    ) -> list[Completion]:
        sampling_params = replace(self.default_sampling_params, **sampling_params)
        self.check_messages_empty(messages)

        if stop_seq:
            sampling_params.stop = [stop_seq] if isinstance(stop_seq, str) else stop_seq
            sampling_params.include_stop_str_in_output = True

        dict_messages = [msg.to_serialized() for msg in messages]
        completions = await self._generate(
            dict_messages, n, sampling_params, verbose, continue_last_message, print_last_message, heartbeat_fn
        )

        return completions

    @retry(
        retry=retry_if_exception(is_retryable_httpx_error),
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=8),
        before_sleep=warn_before_sleep,
        reraise=True,
    )
    async def _generate(
        self,
        messages: list[dict],
        n: int,
        sampling_params: ClientSamplingParams,
        verbose: bool = False,
        continue_last_message: bool = False,
        print_last_message: bool = True,
        heartbeat_fn: Callable | None = None,
    ) -> list[Completion]:
        last_msg = messages[-1]
        if verbose and continue_last_message and print_last_message:
            print_response(last_msg["content"])
        payload = {
            "model": self.model,
            "messages": messages,
            "continue_last_message": continue_last_message,
        } | asdict(sampling_params)

        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            # Always stream responses
            completions = [Completion("", truncated=True) for _ in range(n)]
            async with client.stream("POST", "/generate", json=payload) as resp:
                await my_raise_for_status(resp)
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    outputs = data["outputs"]

                    output_0 = None
                    for output in outputs:
                        i = output["index"]
                        if i == 0:
                            output_0 = output
                        completions[i] += output["delta"]
                        completions[i].truncated = output["truncated"]
                    if verbose and output_0 is not None:
                        print_response(output_0["delta"])

                    # Heartbeat after every streamed generation step (token/chunk)
                    if heartbeat_fn is not None:
                        heartbeat_fn()

        if verbose:
            print_response("\n", color=Colors.DEFAULT)

        return completions

async def my_raise_for_status(resp: httpx.Response):
    if resp.status_code < 400:
        return resp

    try:
        content = await resp.aread()
        error_data = json.loads(content.decode("utf-8"))
        error_message = error_data.get("error", "Unknown server error")
    except ValueError:
        error_message = resp.text or "Non-JSON error response"

    raise ServerError(f"Request failed with status {resp.status_code}: {error_message}")

class MockClient(BaseClient[BaseSamplingParams]):
    """A mock client that returns a sequence of responses, or a default response if the sequence is empty."""

    responses: list[Completion | str]

    def __init__(self, responses: Sequence[Completion | str]):
        self.responses = list(responses)
        self.stored_messages = []
        self.model = "mock"
        self.default_sampling_params = BaseSamplingParams()

    async def _call_impl(
        self,
        messages: Sequence[Message],
        *,
        n: int,
        stop_seq: list[str] | str | None,
        verbose: bool,
        continue_last_message: bool,
        print_last_message: bool,
        heartbeat_fn: Callable | None,
        **sampling_params,
    ) -> list[Completion]:
        if n > 1:
            raise NotImplementedError("Sampling multiple completions (n > 1) is not supported for mock client.")

        sampling_params = replace(self.default_sampling_params, **sampling_params)

        last_msg = messages[-1]
        if verbose and continue_last_message and print_last_message:
            print_response(last_msg.content)

        if not self.responses:
            raise ValueError("No responses left in the mock client")

        content = self.responses.pop(0)
        if stop_seq:
            if isinstance(stop_seq, str):
                stop_seq = [stop_seq]
            for stop_str in stop_seq:
                index = content.find(stop_str)
                if index != -1:
                    content = content[:index] + stop_str
        truncated = getattr(content, "truncated", False)
        completion = Completion(content, truncated=truncated)

        if verbose:
            print_response(completion)
            print_response("\n", color=Colors.DEFAULT)

        self.stored_messages.append({"messages": messages, "response": content})

        return [completion]



def get_client(model: str, **kwargs) -> BaseClient:
    if model == "mock":
        return MockClient(**kwargs)
    else:
        return Client(model, **kwargs)
