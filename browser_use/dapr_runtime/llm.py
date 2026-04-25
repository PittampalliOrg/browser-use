"""LLM selection for the browser-use durable runtime."""

from __future__ import annotations

import os
from typing import Any

from browser_use import ChatAnthropic, ChatGoogle, ChatOpenAI
from browser_use.llm.base import BaseChatModel

DEFAULT_MODEL_SPEC = (
	os.environ.get('BROWSER_USE_AGENT_DEFAULT_MODEL_SPEC')
	or os.environ.get('BROWSER_USE_AGENT_DEFAULT_MODEL')
	or 'openai/gpt-4.1-mini'
)


def build_llm(agent_config: dict[str, Any]) -> BaseChatModel:
	model_spec = str(agent_config.get('modelSpec') or DEFAULT_MODEL_SPEC).strip()
	if not model_spec:
		raise ValueError('modelSpec is required for browser-use durable runtime')

	raw_model = model_spec.split('/', 1)[1] if '/' in model_spec else model_spec
	lowered = model_spec.lower()
	if lowered.startswith('anthropic/') or raw_model.startswith('claude'):
		return ChatAnthropic(model=raw_model)
	if lowered.startswith('google/') or raw_model.startswith('gemini'):
		return ChatGoogle(model=raw_model)
	if lowered.startswith('openai/') or raw_model.startswith(('gpt', 'o1', 'o3', 'o4')):
		return ChatOpenAI(model=raw_model)
	if lowered.startswith(('browser-use/', 'browser_use/', 'bu/')) or raw_model.startswith(('browser-use', 'bu-')):
		raise ValueError(
			f"Unsupported browser-use cloud model {model_spec!r}. "
			'Use a regular model such as openai/gpt-4.1-mini, anthropic/claude-sonnet-4-6, or google/gemini-3.1-pro.'
		)
	raise ValueError(
		f"Unsupported modelSpec {model_spec!r}. "
		'Use a regular model provider prefix: openai/, anthropic/, or google/.'
	)
