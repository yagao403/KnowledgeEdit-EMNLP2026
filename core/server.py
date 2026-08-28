"""Paper-only vLLM server for the custom ``core.client`` protocol.

The retained experiments need text generation with exact repository-specific
message tokenization and support for continuing a partial assistant message.
This server intentionally exposes only health checking and generation.

The implementation targets the vLLM 0.11.x V1 API used by the paper runs.
Install vLLM separately with a wheel matching the host's CUDA or ROCm stack.
"""

from __future__ import annotations

from argparse import Namespace
import asyncio
from collections.abc import AsyncGenerator
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse


# vLLM reads this setting during import. The retained paper launch used V1.
os.environ["VLLM_USE_V1"] = "1"

from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.entrypoints.launcher import serve_http  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402
from vllm.logger import init_logger  # noqa: E402
from vllm.sampling_params import RequestOutputKind, SamplingParams  # noqa: E402
from vllm.usage.usage_lib import UsageContext  # noqa: E402
from vllm.utils import FlexibleArgumentParser, random_uuid  # noqa: E402
from vllm.v1.engine.async_llm import AsyncLLM  # noqa: E402
from vllm.version import __version__ as VLLM_VERSION  # noqa: E402

from core import FAMILY_MAP, S2_MODEL_NAME  # noqa: E402
from core.message import Message  # noqa: E402
from core.tokenizer import BaseTokenizer  # noqa: E402


TIMEOUT_KEEP_ALIVE = 5

logger = init_logger("knowledge_edit.server")
app = FastAPI()
engine: AsyncLLM | None = None
tokenizer: VLLMTokenizer | None = None


class VLLMTokenizer(BaseTokenizer):
    """Adapt a vLLM tokenizer to the repository's text tokenizer API."""

    def __init__(self, wrapped: Any):
        tokenizer_id = str(wrapped.name_or_path)
        if tokenizer_id not in FAMILY_MAP:
            raise ValueError(
                f"Tokenizer {tokenizer_id!r} is not supported. "
                "Pass --tokenizer with an id listed in core/model_map.tsv."
            )
        self.model_family = FAMILY_MAP[tokenizer_id]
        if self.model_family == "qwen-vl":
            raise ValueError("The paper server supports text models only.")
        self.tokenizer = wrapped

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return self.tokenizer.encode(text, add_special_tokens=False)

    def batch_encode(self, texts: list[str]) -> list[list[int]]:
        return self.tokenizer(texts, add_special_tokens=False)["input_ids"]

    def decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


def _engine_model_id() -> str:
    assert engine is not None
    return str(engine.model_config.model).rstrip("/")


def _accepted_model_names(model_id: str) -> set[str]:
    """Accept a registered nickname or a merged model directory name."""
    names = {model_id}
    if short_name := S2_MODEL_NAME.get(model_id):
        names.add(short_name)
    model_path = Path(model_id)
    if model_path.is_absolute() or model_path.exists():
        names.add(model_path.name)
    return names


def _model_error(requested_model: str | None) -> str:
    if requested_model is None:
        return ""
    accepted = _accepted_model_names(_engine_model_id())
    if requested_model.rstrip("/") in accepted:
        return ""
    names = ", ".join(sorted(accepted))
    return f"Requested model {requested_model!r} does not match the served model ({names})."


@app.get("/health")
async def health() -> Response:
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Stream newline-delimited generation deltas for ``core.Client``."""
    request_dict = await request.json()
    try:
        return await _generate(request_dict)
    except Exception as exc:
        logger.exception("Generation request failed")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": str(exc)})


async def _generate(request_dict: dict[str, Any]) -> Response:
    assert engine is not None and tokenizer is not None

    requested_model = request_dict.pop("model", None)
    serialized_messages = request_dict.pop("messages")
    continue_last_message = request_dict.pop("continue_last_message", False)
    error = _model_error(requested_model)
    if error:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": error})

    messages = [Message.from_serialized(message) for message in serialized_messages]
    token_ids = tokenizer.tokenize_for_inference(messages, continue_last_message)
    prompt = TokensPrompt(prompt_token_ids=token_ids)
    sampling_params = SamplingParams(**request_dict)
    eos_token_id = tokenizer.tokenizer.eos_token_id
    if sampling_params.stop_token_ids is None and eos_token_id is not None:
        sampling_params.stop_token_ids = [eos_token_id]
    sampling_params.output_kind = RequestOutputKind.DELTA

    request_id = random_uuid()
    results = engine.generate(prompt, sampling_params, request_id)

    async def stream_results() -> AsyncGenerator[bytes, None]:
        try:
            async for request_output in results:
                outputs = [
                    {
                        "index": output.index,
                        "delta": output.text,
                        "truncated": output.finish_reason != "stop",
                    }
                    for output in request_output.outputs
                ]
                yield (json.dumps({"outputs": outputs}) + "\n").encode("utf-8")
        except asyncio.CancelledError:
            maybe_awaitable = engine.abort(request_id)
            if asyncio.iscoroutine(maybe_awaitable):
                await maybe_awaitable
            raise

    return StreamingResponse(stream_results(), media_type="application/x-ndjson")


async def init_app(args: Namespace) -> FastAPI:
    global engine, tokenizer

    engine_args = AsyncEngineArgs.from_cli_args(args)
    config = engine_args.create_engine_config(usage_context=UsageContext.API_SERVER)
    engine = AsyncLLM.from_vllm_config(
        vllm_config=config,
        usage_context=UsageContext.API_SERVER,
        disable_log_requests=engine_args.disable_log_requests,
        disable_log_stats=engine_args.disable_log_stats,
        client_addresses=None,
        client_index=0,
    )
    app.state.engine_client = engine
    app.state.enable_server_load_tracking = False
    tokenizer = VLLMTokenizer(await engine.get_tokenizer())
    return app


async def run_server(args: Namespace) -> None:
    logger.info("vLLM API server version %s", VLLM_VERSION)
    application = await init_app(args)
    shutdown_task = await serve_http(
        application,
        sock=None,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
    await shutdown_task


def build_cli_parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    return AsyncEngineArgs.add_cli_args(parser)


if __name__ == "__main__":
    asyncio.run(run_server(build_cli_parser().parse_args()))
