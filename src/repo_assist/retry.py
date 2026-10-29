"""Retry model requests that the provider rate-limits or fails transiently.

The Gemini free tier allows 15 requests a minute, and one agent answer can need more
than that (tool steps plus grounding retries). A rate-limited request is retried after
the delay the server asks for. A daily-quota error is not retried: waiting won't help.
"""

import asyncio
import re

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

MAX_ATTEMPTS = 6
RETRYABLE = {429, 500, 502, 503, 504}


def is_daily_quota(exc: ModelHTTPError) -> bool:
    return exc.status_code == 429 and "PerDay" in str(exc.body or exc)


def retry_delay(exc: ModelHTTPError, attempt: int) -> float | None:
    """Seconds to wait before retrying `exc`, or None if it shouldn't be retried."""
    if exc.status_code not in RETRYABLE or is_daily_quota(exc):
        return None
    body = str(exc.body or exc)
    match = re.search(r"retry in ([\d.]+)s", body) or re.search(r"'retryDelay': '(\d+)s'", body)
    return float(match.group(1)) + 1 if match else min(15 * attempt, 90)


class RetryingModel(WrapperModel):
    async def request(self, messages: list[ModelMessage], model_settings: ModelSettings | None,
                      model_request_parameters: ModelRequestParameters) -> ModelResponse:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return await self.wrapped.request(messages, model_settings, model_request_parameters)
            except ModelHTTPError as exc:
                delay = retry_delay(exc, attempt)
                if delay is None or attempt == MAX_ATTEMPTS:
                    raise
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")
