"""Record/replay cache for LLM calls, so evaluation reruns are reproducible and free.

Wraps any pydantic-ai model. A request whose messages, tools and settings match an
earlier one returns the recorded response instead of calling the provider. With a
free tier capped at a few hundred requests a day, this is what makes it practical
to resume interrupted runs, re-score answers after a rubric fix, or re-run one
variant without paying again for the others.
"""

import hashlib
import json
import sqlite3
from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

# Fields that change on every call without changing what the model is asked.
_VOLATILE = {"timestamp", "run_id", "conversation_id", "provider_response_id", "provider_details", "id", "metadata"}


def _canonical(value):
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in sorted(value.items()) if k not in _VOLATILE}
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    return value


def request_key(model_name: str, messages: list[ModelMessage], settings: ModelSettings | None,
                params: ModelRequestParameters) -> str:
    tools = [(t.name, t.description, t.parameters_json_schema)
             for t in [*params.function_tools, *params.output_tools]]
    payload = {
        "model": model_name,
        "messages": _canonical(json.loads(ModelMessagesTypeAdapter.dump_json(messages))),
        "settings": _canonical(dict(settings or {})),
        "tools": tools,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class CachedModel(WrapperModel):
    def __init__(self, wrapped: Model | str, path: Path):
        super().__init__(wrapped)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, response TEXT NOT NULL)")
        self.hits = 0
        self.misses = 0

    async def request(self, messages: list[ModelMessage], model_settings: ModelSettings | None,
                      model_request_parameters: ModelRequestParameters) -> ModelResponse:
        key = request_key(self.wrapped.model_name, messages, model_settings, model_request_parameters)
        row = self._db.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        if row is not None:
            self.hits += 1
            return ModelMessagesTypeAdapter.validate_json(row[0])[0]
        response = await self.wrapped.request(messages, model_settings, model_request_parameters)
        self._db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?)",
                         (key, ModelMessagesTypeAdapter.dump_json([response]).decode()))
        self._db.commit()
        self.misses += 1
        return response
