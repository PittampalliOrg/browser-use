"""Dapr durable runtime host for browser-use."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from google.protobuf import wrappers_pb2

from browser_use import Agent, Browser, Tools
from browser_use.agent.views import AgentHistoryList
from browser_use.dapr_runtime.browserstation import BrowserstationClient, BrowserstationSessionState
from browser_use.dapr_runtime.llm import build_llm as _build_llm
from browser_use.dapr_runtime.models import DurableSessionState, has_persisted_session_history
from browser_use.mcp.client import MCPServerConfig, close_mcp_clients, register_mcp_server_configs_to_tools

try:
	import durabletask.internal.orchestrator_service_pb2 as taskhub_pb
	import durabletask.internal.orchestrator_service_pb2_grpc as taskhub_pb_grpc
	import grpc
	from dapr_agents.agents.configs import AgentExecutionConfig, AgentStateConfig
	from dapr_agents.agents.durable import DurableAgent
	from dapr_agents.llm.dapr import DaprChatClient
	from dapr_agents.storage.daprstores.stateservice import StateStoreService
	from dapr_agents.workflow.decorators import message_router, workflow_entry
	from dapr_agents.workflow.runners import AgentRunner
	from dapr_agents.agents.schemas import TriggerAction
except ImportError as exc:  # pragma: no cover - import error is surfaced at startup time
	raise RuntimeError('browser-use durable runtime requires dapr-agents and dapr-ext-workflow') from exc

logger = logging.getLogger(__name__)

SERVICE_NAME = os.environ.get('BROWSER_USE_AGENT_SERVICE_NAME', 'browser-use-agent')
STATE_STORE_NAME = os.environ.get('BROWSER_USE_AGENT_STATE_STORE', 'dapr-agent-py-statestore')
DEFAULT_LLM_COMPONENT = os.environ.get('DAPR_LLM_COMPONENT_DEFAULT', 'llm-openai-gpt5')
DEFAULT_MAX_TURNS = int(os.environ.get('BROWSER_USE_AGENT_DEFAULT_MAX_TURNS', '40'))
DEFAULT_STEP_TIMEOUT = int(os.environ.get('BROWSER_USE_AGENT_STEP_TIMEOUT_SECONDS', '180'))



def _dapr_http_base() -> str:
	return f"http://{os.environ.get('DAPR_HOST', '127.0.0.1')}:{os.environ.get('DAPR_HTTP_PORT', '3500')}"


def _dapr_grpc_target() -> str:
	return f"{os.environ.get('DAPR_HOST', '127.0.0.1')}:{os.environ.get('DAPR_GRPC_PORT', '50001')}"


def _taskhub_call(method: str, request: Any) -> Any:
	stub = taskhub_pb_grpc.TaskHubSidecarServiceStub(grpc.insecure_channel(_dapr_grpc_target()))
	return getattr(stub, method)(request, timeout=float(os.environ.get('TASKHUB_RPC_TIMEOUT_SECONDS', '15')))


def _state_key(session_id: str) -> str:
	return f'{SERVICE_NAME}:session:{session_id}'


def _read_json_state(key: str) -> dict[str, Any] | None:
	encoded_key = urllib.parse.quote(key, safe='')
	url = (
		f"{_dapr_http_base()}/v1.0/state/{urllib.parse.quote(STATE_STORE_NAME, safe='')}/{encoded_key}"
		f'?metadata.partitionKey={encoded_key}'
	)
	try:
		with urllib.request.urlopen(url, timeout=5) as response:
			raw = response.read().decode('utf-8')
	except Exception:
		return None
	try:
		value = json.loads(raw)
		return value if isinstance(value, dict) else None
	except json.JSONDecodeError:
		return None


def _write_json_state(key: str, value: dict[str, Any]) -> None:
	encoded_key = urllib.parse.quote(key, safe='')
	url = f"{_dapr_http_base()}/v1.0/state/{urllib.parse.quote(STATE_STORE_NAME, safe='')}"
	payload = json.dumps(
		[
			{
				'key': key,
				'value': json.dumps(value),
				'metadata': {'partitionKey': encoded_key},
			}
		]
	).encode('utf-8')
	request = urllib.request.Request(
		url,
		data=payload,
		headers={'Content-Type': 'application/json'},
		method='POST',
	)
	with urllib.request.urlopen(request, timeout=5):
		return


def _delete_state(key: str) -> None:
	encoded_key = urllib.parse.quote(key, safe='')
	url = (
		f"{_dapr_http_base()}/v1.0/state/{urllib.parse.quote(STATE_STORE_NAME, safe='')}/{encoded_key}"
		f'?metadata.partitionKey={encoded_key}'
	)
	request = urllib.request.Request(url, method='DELETE')
	with urllib.request.urlopen(request, timeout=5):
		return


def _load_session_state(session_id: str) -> DurableSessionState:
	value = _read_json_state(_state_key(session_id))
	if not value:
		return DurableSessionState()
	return DurableSessionState.model_validate(value)


def _save_session_state(session_id: str, state: DurableSessionState) -> None:
	_write_json_state(_state_key(session_id), state.model_dump(mode='json'))


def _publish_session_event(session_id: str, event_type: str, data: dict[str, Any] | None = None) -> None:
	"""Persist a session event into workflow-builder, best-effort.

	Stamps ``traceId`` + ``spanId`` of the currently-active OTEL span onto
	the envelope's ``data`` payload so the Timeline UI can deep-link each
	event row into Phoenix / ClickHouse without needing a separate
	correlation step. Mirrors dapr-agent-py's
	``services/dapr-agent-py/src/event_publisher.py::publish_session_event``
	behavior (lines 277-291 in that file).
	"""
	internal_token = os.environ.get('INTERNAL_API_TOKEN', '').strip()
	if not internal_token or not session_id:
		return

	payload = dict(data or {})
	# Stamp trace context so the UI can deep-link this event to its Phoenix
	# trace. Best-effort — returns (None, None) when telemetry isn't wired.
	try:
		from browser_use.dapr_runtime.telemetry import get_current_trace_context

		trace_id, span_id = get_current_trace_context()
		if trace_id:
			payload.setdefault('traceId', trace_id)
		if span_id:
			payload.setdefault('spanId', span_id)
	except Exception:
		pass

	app_id = os.environ.get('WORKFLOW_BUILDER_APP_ID', 'workflow-builder')
	url = (
		f"{_dapr_http_base()}/v1.0/invoke/{urllib.parse.quote(app_id, safe='')}"
		f'/method/api/internal/sessions/{urllib.parse.quote(session_id, safe="")}/events/ingest'
	)
	body = {
		'type': event_type,
		'data': payload,
		'producerId': os.environ.get('AGENT_SLUG', SERVICE_NAME),
		'producerEpoch': os.environ.get('HOSTNAME', SERVICE_NAME),
	}
	request = urllib.request.Request(
		url,
		data=json.dumps(body).encode('utf-8'),
		headers={
			'Content-Type': 'application/json',
			'X-Internal-Token': internal_token,
		},
		method='POST',
	)
	try:
		with urllib.request.urlopen(request, timeout=5):
			return
	except Exception:
		logger.debug('Failed to publish session event %s for %s', event_type, session_id, exc_info=True)


def _compose_turn_task(events: list[dict[str, Any]]) -> str:
	parts: list[str] = []
	for event in events:
		if str(event.get('type') or '') != 'user.message':
			continue
		content = event.get('content') or event.get('data', {}).get('content') or []
		for block in content:
			if isinstance(block, dict) and block.get('type') == 'text':
				text = str(block.get('text') or '').strip()
				if text:
					parts.append(text)
	return '\n\n'.join(parts)


def _freeze_session_child_input(
	*,
	session_id: str,
	agent_cfg: dict[str, Any],
	vault_ids: list[Any],
	db_execution_id: str,
	turn: int,
	task: str,
	raw_message: dict[str, Any],
) -> dict[str, Any]:
	return {
		'task': task,
		'prompt': task,
		'sessionId': session_id,
		'executionId': db_execution_id,
		'dbExecutionId': db_execution_id,
		'workflowExecutionId': db_execution_id,
		'workflowId': raw_message.get('workflowId'),
		'nodeId': raw_message.get('nodeId'),
		'agentConfig': agent_cfg,
		'vaultIds': vault_ids,
		'sandboxName': raw_message.get('sandboxName'),
		'workspaceRef': raw_message.get('workspaceRef'),
		'cwd': raw_message.get('cwd'),
		'_session_turn': turn,
		'_message_metadata': {
			'executionId': db_execution_id,
			'sessionId': session_id,
			'turn': turn,
		},
	}


def _build_agent_instruction_block(agent_config: dict[str, Any]) -> str | None:
	lines: list[str] = []
	role = str(agent_config.get('role') or '').strip()
	goal = str(agent_config.get('goal') or '').strip()
	if role:
		lines.append(f'Role: {role}')
	if goal:
		lines.append(f'Goal: {goal}')
	for instruction in agent_config.get('instructions') or []:
		text = str(instruction).strip()
		if text:
			lines.append(f'Instruction: {text}')
	for guideline in agent_config.get('styleGuidelines') or []:
		text = str(guideline).strip()
		if text:
			lines.append(f'Style: {text}')
	return '\n'.join(lines) if lines else None


def _artifact_capture_config(agent_config: dict[str, Any]) -> dict[str, bool]:
	raw = agent_config.get('browserArtifacts')
	if not isinstance(raw, dict):
		return {'screenshots': False, 'video': False}
	return {
		'screenshots': bool(raw.get('screenshots')),
		'video': bool(raw.get('video')),
	}


def _artifact_root(session_id: str, turn: int) -> Path:
	base_dir = Path(tempfile.gettempdir()) / 'browser-use-dapr-artifacts'
	return base_dir / session_id / f'turn-{turn}'


def _file_to_base64(path: Path) -> str:
	return base64.b64encode(path.read_bytes()).decode('utf-8')


def _thumbnail_screenshot(
	source_b64: str | None = None,
	source_path: Path | None = None,
	max_width: int = 720,
	quality: int = 70,
) -> tuple[str, str] | None:
	"""Produce a (media_type, base64) JPEG thumbnail suitable for inline Timeline rendering.

	Accepts a PNG/JPEG either as base64 (from ``BrowserStateSummary.screenshot``)
	or as a disk path (from ``AgentHistory.state.screenshot_path``). Returns
	None on any error so the emit path can skip the image gracefully — an
	event with the text content still goes through.

	Target ~15-40 KB base64 to keep session_events rows + SSE frames small.
	"""
	try:
		from PIL import Image

		if source_b64 is not None:
			raw = base64.b64decode(source_b64)
		elif source_path is not None and source_path.exists():
			raw = source_path.read_bytes()
		else:
			return None
		img = Image.open(io.BytesIO(raw))
		img.thumbnail((max_width, max_width * 2))
		out = io.BytesIO()
		img.convert('RGB').save(out, format='JPEG', quality=quality, optimize=True)
		return ('image/jpeg', base64.b64encode(out.getvalue()).decode('ascii'))
	except Exception:
		logger.debug('screenshot thumbnail failed', exc_info=True)
		return None


def _step_timestamp_iso(step_metadata: Any) -> str | None:
	if step_metadata is None:
		return None
	step_end_time = getattr(step_metadata, 'step_end_time', None)
	if not isinstance(step_end_time, (int, float)):
		return None
	return datetime.fromtimestamp(step_end_time, tz=timezone.utc).isoformat()


def _build_browser_artifact_steps(history: AgentHistoryList) -> list[dict[str, Any]]:
	steps: list[dict[str, Any]] = []
	for index, item in enumerate(history.history, start=1):
		actions: list[str] = []
		goal: str | None = None
		if item.model_output is not None:
			goal = getattr(item.model_output.current_state, 'next_goal', None)
			for action in item.model_output.action:
				action_dump = action.model_dump(exclude_none=True, mode='json')
				if action_dump:
					actions.append(next(iter(action_dump.keys())))
		error_text = next((result.error for result in item.result if result.error), None)
		step: dict[str, Any] = {
			'id': f'step-{index}',
			'label': f'Step {index}',
			'url': item.state.url or '',
			'title': item.state.title or '',
			'capturedAt': _step_timestamp_iso(item.metadata),
			'status': 'failed' if error_text else 'completed',
		}
		if actions:
			step['action'] = ', '.join(actions)
		if goal:
			step['goal'] = str(goal)
		if error_text:
			step['error'] = error_text
		steps.append(step)
	return steps


def _publish_browser_artifact(
	session_id: str,
	payload: dict[str, Any],
	history: AgentHistoryList,
	artifact_root: Path,
) -> None:
	workflow_execution_id = str(
		payload.get('workflowExecutionId') or payload.get('dbExecutionId') or payload.get('executionId') or ''
	).strip()
	workflow_id = str(payload.get('workflowId') or '').strip()
	node_id = str(payload.get('nodeId') or '').strip()
	if not workflow_execution_id or not workflow_id or not node_id:
		return

	internal_token = os.environ.get('INTERNAL_API_TOKEN', '').strip()
	if not internal_token:
		return

	screenshot_paths = [Path(path) for path in history.screenshot_paths(return_none_if_not_screenshot=False)]
	screenshots: list[dict[str, Any]] = []
	for index, screenshot_path in enumerate(screenshot_paths, start=1):
		if not screenshot_path.exists():
			continue
		screenshots.append(
			{
				'payloadBase64': _file_to_base64(screenshot_path),
				'contentType': 'image/png',
				'stepId': f'step-{index}',
				'label': f'Step {index} Screenshot',
			}
		)

	assets: list[dict[str, Any]] = []
	for video_path in sorted(artifact_root.glob('recordings/*.mp4')):
		if not video_path.exists():
			continue
		assets.append(
			{
				'kind': 'video',
				'payloadBase64': _file_to_base64(video_path),
				'contentType': 'video/mp4',
				'fileName': video_path.name,
				'label': 'Browser session recording',
			}
		)

	body = {
		'workflowExecutionId': workflow_execution_id,
		'workflowId': workflow_id,
		'nodeId': node_id,
		'workspaceRef': payload.get('workspaceRef'),
		'baseUrl': next((url for url in history.urls() if isinstance(url, str) and url.strip()), ''),
		'status': 'completed' if history.is_successful() is not False else 'failed',
		'metadata': {
			'sessionId': session_id,
			'agentRuntime': payload.get('agentRuntime') or 'browser-use-agent',
			'turn': payload.get('_session_turn') or 1,
		},
		'steps': _build_browser_artifact_steps(history),
		'screenshots': screenshots,
		'assets': assets,
	}

	app_id = os.environ.get('WORKFLOW_BUILDER_APP_ID', 'workflow-builder')
	url = (
		f"{_dapr_http_base()}/v1.0/invoke/{urllib.parse.quote(app_id, safe='')}"
		'/method/api/internal/browser-artifacts'
	)
	request = urllib.request.Request(
		url,
		data=json.dumps(body).encode('utf-8'),
		headers={
			'Content-Type': 'application/json',
			'X-Internal-Token': internal_token,
		},
		method='POST',
	)
	try:
		with urllib.request.urlopen(request, timeout=20):
			return
	except Exception:
		logger.debug('Failed to publish browser artifact for session %s', session_id, exc_info=True)


async def _run_browser_use_turn_async(payload: dict[str, Any]) -> dict[str, Any]:
	session_id = str(payload.get('sessionId') or '').strip()
	task = str(payload.get('task') or payload.get('prompt') or '').strip()
	if not session_id:
		raise ValueError('sessionId is required')
	if not task:
		raise ValueError('task is required')

	agent_config = payload.get('agentConfig') if isinstance(payload.get('agentConfig'), dict) else {}
	max_steps = int(agent_config.get('maxTurns') or payload.get('maxTurns') or DEFAULT_MAX_TURNS)
	session_state = _load_session_state(session_id)
	browser_client = BrowserstationClient.from_env()
	artifact_capture = _artifact_capture_config(agent_config)
	turn = int(payload.get('_session_turn') or 1)
	artifact_root = _artifact_root(session_id, turn)
	recording_dir = artifact_root / 'recordings'
	if artifact_capture['video']:
		recording_dir.mkdir(parents=True, exist_ok=True)

	if session_state.browserstation is None:
		session_state.browserstation = await browser_client.create_browser()
		_save_session_state(session_id, session_state)

	browser = Browser(
		cdp_url=session_state.browserstation.websocket_url,
		is_local=False,
		keep_alive=True,
		record_video_dir=recording_dir if artifact_capture['video'] else None,
	)
	tools = Tools()
	mcp_clients = []
	mcp_servers = [MCPServerConfig.from_workflow_builder(cfg) for cfg in (agent_config.get('mcpServers') or []) if isinstance(cfg, dict)]
	if mcp_servers:
		mcp_clients = await register_mcp_server_configs_to_tools(tools, mcp_servers)

	extend_system_message = _build_agent_instruction_block(agent_config)
	injected_state = session_state if has_persisted_session_history(session_state) else None
	latest_step_text: dict[str, str | None] = {'value': None}
	# Stable id per (step, action) so agent.tool_use and agent.tool_result can
	# be linked by the Timeline's grouping logic. Populated in _on_step,
	# consumed when emitting tool_result (from the next _on_step for prior
	# steps, and from _on_done for the terminal step).
	emitted_action_ids: list[tuple[int, list[str]]] = []
	# Tracks the last step index whose tool_result events have been emitted.
	# _on_step drains everything up to (but not including) the current step's
	# index; _on_done drains the rest (typically just the terminal step).
	last_emitted_result_step: dict[str, int] = {'value': 0}

	def _emit_tool_result_for_step(
		history_idx: int, history_item: Any, post_action_image: tuple[str, str] | None
	) -> None:
		"""Emit one ``agent.tool_result`` per ActionResult in this step.

		`post_action_image` is a (media_type, base64) thumbnail of the page
		AFTER this step's actions executed, to be attached as an Anthropic
		image content block so the Timeline can render a live filmstrip.
		When None, the event still goes out with text-only content.
		"""
		if history_idx >= len(emitted_action_ids):
			return
		_step_idx, action_ids = emitted_action_ids[history_idx]
		results = list(getattr(history_item, 'result', None) or [])
		state = getattr(history_item, 'state', None)
		url = getattr(state, 'url', None) if state is not None else None
		for action_idx, res in enumerate(results):
			tool_use_id = action_ids[action_idx] if action_idx < len(action_ids) else f'{_step_idx}-{action_idx}'
			extracted = getattr(res, 'extracted_content', None)
			error = getattr(res, 'error', None)
			output_text = str(extracted) if extracted is not None else ''
			content: list[dict[str, Any]] = []
			if output_text:
				content.append({'type': 'text', 'text': output_text})
			# Attach the thumbnail to the LAST action of the step only — one
			# screenshot represents the post-step page state, regardless of
			# how many actions fired. Avoids N duplicate thumbnails per step.
			if post_action_image is not None and action_idx == len(results) - 1:
				media_type, b64 = post_action_image
				content.append(
					{
						'type': 'image',
						'source': {'type': 'base64', 'media_type': media_type, 'data': b64},
					}
				)
			payload_out: dict[str, Any] = {
				'tool_use_id': tool_use_id,
				'output': output_text,
				'stepNumber': _step_idx,
			}
			if url:
				payload_out['url'] = str(url)
			if content:
				payload_out['content'] = content
			if error:
				payload_out['error'] = str(error)
				payload_out['is_error'] = True
			_publish_session_event(session_id, 'agent.tool_result', payload_out)

	async def _on_step(browser_state: Any, model_output: Any, step_info: Any) -> None:
		step_idx = int(getattr(step_info, 'step_number', 0) or len(emitted_action_ids) + 1)

		# Flush pending tool_results for every prior step using the CURRENT
		# observation's screenshot as the post-action state. browser-use
		# doesn't always populate agent.state.history at _on_step time, so we
		# emit tool_result purely from the tool_use_ids tracked in
		# emitted_action_ids. The `output` field is left empty since the
		# ActionResult isn't yet accessible — this is a Timeline-visibility
		# emission; the final _on_done call re-emits the terminal step with
		# full extracted_content + error info.
		post_action_b64 = getattr(browser_state, 'screenshot', None)
		thumbnail = _thumbnail_screenshot(source_b64=post_action_b64) if post_action_b64 else None
		page_url = getattr(browser_state, 'url', None)
		start = last_emitted_result_step['value']
		for prev_idx in range(start, len(emitted_action_ids)):
			_prev_step, prev_action_ids = emitted_action_ids[prev_idx]
			if not prev_action_ids:
				last_emitted_result_step['value'] = prev_idx + 1
				continue
			is_latest = prev_idx == len(emitted_action_ids) - 1
			# One tool_result per planned action; the thumbnail attaches to
			# the LAST action of the LATEST flushed step only (a single
			# post-step screenshot).
			for action_idx, tool_use_id in enumerate(prev_action_ids):
				attach = (
					thumbnail
					if (is_latest and action_idx == len(prev_action_ids) - 1)
					else None
				)
				content: list[dict[str, Any]] = []
				if attach is not None:
					media_type, b64 = attach
					content.append(
						{
							'type': 'image',
							'source': {'type': 'base64', 'media_type': media_type, 'data': b64},
						}
					)
				payload: dict[str, Any] = {
					'tool_use_id': tool_use_id,
					'output': '',
					'stepNumber': _prev_step,
				}
				if page_url:
					payload['url'] = str(page_url)
				if content:
					payload['content'] = content
				_publish_session_event(session_id, 'agent.tool_result', payload)
			last_emitted_result_step['value'] = prev_idx + 1

		current_state = getattr(model_output, 'current_state', None)

		# agent.thinking — full chain-of-thought from the LLM (when present)
		# or the evaluation_previous_goal as fallback.
		thinking_text = getattr(model_output, 'thinking', None) or getattr(
			current_state, 'evaluation_previous_goal', None
		)
		if thinking_text:
			_publish_session_event(
				session_id,
				'agent.thinking',
				{'thinking': str(thinking_text), 'type': 'text'},
			)

		# agent.message — next_goal, preserved behavior
		goal = getattr(current_state, 'next_goal', None)
		if goal:
			latest_step_text['value'] = str(goal)
			_publish_session_event(
				session_id,
				'agent.message',
				{
					'role': 'assistant',
					'content': [{'type': 'text', 'text': latest_step_text['value']}],
				},
			)

		# agent.tool_use — one event per planned action on this step.
		actions = list(getattr(model_output, 'action', None) or [])
		ids_this_step: list[str] = []
		for action_idx, action in enumerate(actions):
			action_id = f'step-{step_idx}-action-{action_idx}'
			ids_this_step.append(action_id)
			# ActionModel is a dynamic Pydantic model: exactly one field is
			# populated, named after the action. Extract both.
			try:
				dumped = action.model_dump(exclude_none=True) if hasattr(action, 'model_dump') else dict(action or {})
			except Exception:
				dumped = {}
			action_name = next(iter(dumped.keys()), 'unknown') if dumped else 'unknown'
			action_input = dumped.get(action_name) if dumped else {}
			_publish_session_event(
				session_id,
				'agent.tool_use',
				{
					'id': action_id,
					'name': str(action_name),
					'input': action_input if isinstance(action_input, dict) else {'value': action_input},
				},
			)
		emitted_action_ids.append((step_idx, ids_this_step))

	async def _on_done(history_list: 'AgentHistoryList') -> None:
		# Drain any step(s) that completed after the last _on_step emission.
		# Attach the post-action thumbnail to the LAST history item that has
		# a non-empty `result` list — browser-use often emits a trailing
		# history item with an empty result (after an LLM produces a `done`
		# action without a new tool call); inlining the screenshot there
		# would get dropped because `_emit_tool_result_for_step` iterates
		# over results. Finding the last-with-results ensures the thumbnail
		# reaches the Timeline.
		step_items = list(getattr(history_list, 'history', None) or [])
		start = last_emitted_result_step['value']
		logger.info(
			'[tool_result] _on_done draining history %d..%d (total %d items)',
			start,
			len(step_items),
			len(step_items),
		)
		# Resolve the thumbnail once (source: latest state with a screenshot).
		thumbnail: tuple[str, str] | None = None
		for item in reversed(step_items):
			state = getattr(item, 'state', None)
			if state is None:
				continue
			b64: str | None = None
			if hasattr(state, 'get_screenshot'):
				try:
					b64 = state.get_screenshot()
				except Exception as exc:
					logger.warning('[tool_result] get_screenshot() raised: %s', exc)
			if not b64:
				spath = getattr(state, 'screenshot_path', None)
				if spath:
					try:
						p = Path(spath)
						if p.exists():
							b64 = base64.b64encode(p.read_bytes()).decode('utf-8')
					except Exception as exc:
						logger.warning('[tool_result] direct-read fallback failed: %s', exc)
			if b64:
				thumbnail = _thumbnail_screenshot(source_b64=b64)
				if thumbnail is not None:
					break
		logger.info('[tool_result] final thumbnail resolved=%s', thumbnail is not None)

		# Find the last history idx that `_emit_tool_result_for_step` will
		# actually publish for (requires a corresponding emitted_action_ids
		# entry AND a non-empty result list — actions that bypass the
		# register_new_step_callback path don't get emit slots).
		last_with_results = -1
		for i in range(len(step_items) - 1, start - 1, -1):
			if i >= len(emitted_action_ids):
				continue
			if list(getattr(step_items[i], 'result', None) or []):
				last_with_results = i
				break
		logger.info(
			'[tool_result] last_with_results=%d emitted_action_ids=%d',
			last_with_results,
			len(emitted_action_ids),
		)

		for history_idx in range(start, len(step_items)):
			item = step_items[history_idx]
			attach = thumbnail if history_idx == last_with_results else None
			_emit_tool_result_for_step(history_idx, item, attach)
			last_emitted_result_step['value'] = history_idx + 1

		# If NO step in this drain range had results (rare — a pure terminal
		# step with only a done signal), emit a standalone tool_result tied
		# to the LAST emitted_action_ids entry so the thumbnail isn't lost.
		if thumbnail is not None and last_with_results == -1 and emitted_action_ids:
			_step_idx, action_ids = emitted_action_ids[-1]
			fallback_id = action_ids[-1] if action_ids else f'step-{_step_idx}-action-final'
			media_type, b64 = thumbnail
			_publish_session_event(
				session_id,
				'agent.tool_result',
				{
					'tool_use_id': fallback_id,
					'output': '',
					'stepNumber': _step_idx,
					'content': [
						{
							'type': 'image',
							'source': {'type': 'base64', 'media_type': media_type, 'data': b64},
						}
					],
				},
			)

	# Wrap the browser-use run in a span so Phoenix/Tempo shows a single
	# `browser_use.agent.turn` tree per turn, with session.id / workflow
	# attributes that match the ClickHouse filter in
	# src/lib/server/otel/clickhouse.ts:70-92.
	try:
		from browser_use.dapr_runtime.telemetry import get_tracer
	except Exception:
		get_tracer = lambda: None  # noqa: E731

	tracer = get_tracer()
	span_cm = (
		tracer.start_as_current_span(
			'browser_use.agent.turn',
			attributes={
				'session.id': session_id,
				'agent.slug': os.environ.get('AGENT_SLUG', SERVICE_NAME),
				'workflow.execution.id': str(payload.get('workflowExecutionId') or ''),
				'workflow.id': str(payload.get('workflowId') or ''),
				'agent.turn': turn,
			},
		)
		if tracer is not None
		else None
	)

	agent_init_kwargs: dict[str, Any] = {
		'task': task,
		'llm': _build_llm(agent_config),
		'browser': browser,
		'tools': tools,
		'extend_system_message': extend_system_message,
		'llm_timeout': int(agent_config.get('timeoutMinutes') or 90) * 60,
		'step_timeout': DEFAULT_STEP_TIMEOUT,
		'injected_agent_state': injected_state,
		'register_new_step_callback': _on_step,
		'register_done_callback': _on_done,
	}

	# Per-task tunable overrides from the workflow JSON's
	# do[].browser_use_agent.with.agentKwargs block. Whitelist supported
	# keys so a workflow shipping unknown kwargs against an older
	# browser-use version warns instead of crashing the agent.
	_SUPPORTED_AGENT_OVERRIDES = {
		'vision_detail_level',
		'max_actions_per_step',
		'max_history_items',
		'max_failures',
		'use_thinking',
		'use_vision',
		'flash_mode',
		'step_timeout',
		'llm_timeout',
	}
	override = agent_config.get('agentKwargs') or payload.get('agentKwargs') or {}
	if isinstance(override, dict):
		for k, v in override.items():
			if k in _SUPPORTED_AGENT_OVERRIDES:
				agent_init_kwargs[k] = v
				logger.info('Applying agentKwargs override from workflow: %s=%r', k, v)
			else:
				logger.warning(
					'Ignoring unsupported agentKwargs key from workflow: %s (not in whitelist)',
					k,
				)

	agent = Agent(**agent_init_kwargs)

	history: AgentHistoryList | None = None
	result_payload: dict[str, Any] | None = None

	async def _do_run() -> dict[str, Any]:
		nonlocal history
		history = await agent.run(max_steps=max_steps)
		assert history is not None
		session_state_local = DurableSessionState.model_validate(agent.state.model_dump(mode='json'))
		session_state_local.browserstation = session_state_local.browserstation or _load_session_state(session_id).browserstation
		_save_session_state(session_id, session_state_local)
		final_content_local = history.final_result() or latest_step_text['value'] or ''
		if final_content_local:
			_publish_session_event(
				session_id,
				'agent.message',
				{
					'role': 'assistant',
					'content': [{'type': 'text', 'text': str(final_content_local)}],
				},
			)
		# agent.llm_usage — best-effort, aggregated per-turn. Timeline renders
		# it as a small tile. Fields optional; Timeline tolerates absent keys.
		try:
			tokens = getattr(agent.state, 'usage', None) or getattr(agent.state, 'tokens_used', None)
			if tokens is not None and hasattr(tokens, 'model_dump'):
				usage_dump = tokens.model_dump(exclude_none=True)
				if usage_dump:
					_publish_session_event(session_id, 'agent.llm_usage', {'usage': usage_dump})
			elif isinstance(tokens, dict) and tokens:
				_publish_session_event(session_id, 'agent.llm_usage', {'usage': tokens})
		except Exception:
			pass
		return {
			'success': bool(history.is_successful() if history.is_done() else not history.has_errors()),
			'content': str(final_content_local),
			'urls': history.urls(),
			'errors': [error for error in history.errors() if error],
			'sessionId': session_id,
			'turn': turn,
		}

	try:
		if span_cm is not None:
			with span_cm:
				result_payload = await _do_run()
		else:
			result_payload = await _do_run()
		return result_payload
	finally:
		await close_mcp_clients(mcp_clients)
		await agent.close()
		if history is not None and (artifact_capture['screenshots'] or artifact_capture['video']):
			_publish_browser_artifact(session_id, payload, history, artifact_root)


class BrowserUseDurableAgent(DurableAgent):
	"""DurableAgent wrapper that runs browser-use turns as activities."""

	def register_workflows(self, runtime) -> None:
		super().register_workflows(runtime)
		runtime.register_workflow(self.agent_workflow)
		runtime.register_workflow(self.session_workflow)
		runtime.register_activity(self.run_browser_use_turn)
		runtime.register_activity(self.cleanup_session)

	@workflow_entry
	@message_router(message_model=TriggerAction)
	def agent_workflow(self, ctx, message: dict[str, Any]):
		result = yield ctx.call_activity(self.run_browser_use_turn, input=message)
		final_result = result if isinstance(result, dict) else {'content': str(result or '')}
		if bool(message.get('autoTerminateAfterEndTurn')):
			yield ctx.call_activity(
				self.cleanup_session,
				input={
					'sessionId': str(message.get('sessionId') or ''),
					'reason': 'auto_terminate_after_agent_workflow',
				},
			)
		return final_result

	def run_browser_use_turn(self, ctx, payload: dict[str, Any]) -> dict[str, Any]:
		session_id = str(payload.get('sessionId') or '').strip()
		if session_id:
			_publish_session_event(session_id, 'session.status_running', {'turn': payload.get('_session_turn') or 1})
		try:
			result = self._run_asyncio_task(_run_browser_use_turn_async(payload))
		except Exception as exc:
			if session_id:
				_publish_session_event(session_id, 'session.error', {'error': str(exc)[:500]})
			raise
		if session_id:
			_publish_session_event(session_id, 'session.status_idle', {'stop_reason': {'type': 'end_turn'}})
		return result

	def cleanup_session(self, ctx, payload: dict[str, Any]) -> dict[str, Any]:
		session_id = str(payload.get('sessionId') or '').strip()
		if not session_id:
			return {'ok': True}
		state = _load_session_state(session_id)
		if state.browserstation is not None:
			self._run_asyncio_task(BrowserstationClient.from_env().delete_browser(state.browserstation.browser_id))
		try:
			_delete_state(_state_key(session_id))
		except Exception:
			logger.debug('Failed to delete state for session %s', session_id, exc_info=True)
		_publish_session_event(session_id, 'session.status_terminated', {'reason': payload.get('reason') or 'terminated'})
		return {'ok': True}

	@workflow_entry
	def session_workflow(self, ctx, message: dict[str, Any]):
		session_id = str(message.get('sessionId') or '')
		if not session_id:
			raise RuntimeError('session_workflow requires sessionId')

		agent_cfg = message.get('agentConfig') if isinstance(message.get('agentConfig'), dict) else {}
		vault_ids = message.get('vaultIds') or []
		db_execution_id = str(message.get('dbExecutionId') or '')
		pending = list(message.get('initialEvents') or [])
		auto_terminate = bool(message.get('autoTerminateAfterEndTurn'))

		if not ctx.is_replaying:
			_publish_session_event(session_id, 'session.status_rescheduled', {'vaultIds': vault_ids})

		turn_counter = 0
		while True:
			if not pending:
				if not ctx.is_replaying:
					_publish_session_event(session_id, 'session.status_idle', {'stop_reason': {'type': 'end_turn'}})
				batch = yield ctx.wait_for_external_event('session.user_events')
				pending = list((batch or {}).get('events') or [])
				if not pending:
					continue

			if any(str(event.get('type') or '') == 'session.terminate' for event in pending):
				yield ctx.call_activity(self.cleanup_session, input={'sessionId': session_id, 'reason': 'terminate_event'})
				return

			task_text = _compose_turn_task(pending)
			pending = []
			turn_counter += 1
			child_input = _freeze_session_child_input(
				session_id=session_id,
				agent_cfg=agent_cfg,
				vault_ids=vault_ids,
				db_execution_id=db_execution_id,
				turn=turn_counter,
				task=task_text,
				raw_message=message,
			)
			child_result = yield ctx.call_child_workflow(
				self.agent_workflow_name,
				input=child_input,
				instance_id=f'{session_id}:turn-{turn_counter}',
				retry_policy=self._retry_policy,
			)
			result = child_result if isinstance(child_result, dict) else {'content': str(child_result or '')}
			result.setdefault('sessionId', session_id)
			result.setdefault('turn', turn_counter)
			result.setdefault('success', not bool(result.get('error')))
			if auto_terminate:
				yield ctx.call_activity(
					self.cleanup_session,
					input={'sessionId': session_id, 'reason': 'auto_terminate_after_end_turn'},
				)
				return result


agent = BrowserUseDurableAgent(
	name=SERVICE_NAME,
	role='Browser Use Durable Agent',
	goal='Execute browser automation tasks against Browserstation-backed browsers',
	instructions=['Run browser tasks with browser-use and preserve session browser state between turns.'],
	style_guidelines=['Be direct and concise.'],
	llm=DaprChatClient(component_name=DEFAULT_LLM_COMPONENT),
	execution=AgentExecutionConfig(max_iterations=1),
	state=AgentStateConfig(store=StateStoreService(store_name=STATE_STORE_NAME)),
	agent_metadata={
		'service': SERVICE_NAME,
		'stateStore': STATE_STORE_NAME,
		'instancesEndpoint': '/agent/instances',
	},
)

runner = AgentRunner()


@asynccontextmanager
async def lifespan(_app: FastAPI):
	# Initialize OpenTelemetry providers (traces, metrics, logs). No-op when
	# OTEL_EXPORTER_OTLP_ENDPOINT is unset; safe to call on every pod boot.
	try:
		from browser_use.dapr_runtime.telemetry import init_telemetry, shutdown_telemetry

		init_telemetry()
	except Exception as exc:  # noqa: BLE001
		logger.warning('Telemetry init failed: %s', exc)
		shutdown_telemetry = None  # type: ignore[assignment]

	logger.info('%s starting', SERVICE_NAME)
	try:
		yield
	finally:
		logger.info('%s shutting down', SERVICE_NAME)
		runner.shutdown(agent)
		if shutdown_telemetry is not None:
			try:
				shutdown_telemetry()
			except Exception:
				logger.debug('shutdown_telemetry failed', exc_info=True)


app = FastAPI(
	title=SERVICE_NAME,
	description='Dapr durable runtime host for browser-use',
	version='0.1.0',
	lifespan=lifespan,
)

runner.serve(agent, app=app, port=8002)


@app.get('/healthz')
async def health_check() -> dict[str, Any]:
	return {'status': 'healthy', 'service': SERVICE_NAME}


@app.get('/readyz')
async def ready_check() -> dict[str, Any]:
	return {'status': 'ready', 'service': SERVICE_NAME}


@app.post('/internal/sessions/spawn')
def spawn_session_endpoint(request: dict[str, Any]) -> dict[str, Any]:
	instance_id = str(request.get('instanceId') or '').strip()
	if not instance_id:
		raise HTTPException(status_code=400, detail='instanceId is required')
	payload = request.get('payload') or {}
	create_request = taskhub_pb.CreateInstanceRequest(
		instanceId=instance_id,
		name='session_workflow',
		input=wrappers_pb2.StringValue(value=json.dumps(payload)),
	)
	try:
		_taskhub_call('StartInstance', create_request)
	except Exception as exc:
		message = str(exc)
		if 'already exists' not in message.lower() and 'ALREADY_EXISTS' not in message:
			raise HTTPException(status_code=500, detail=f'StartInstance failed: {message}') from exc
	return {'instanceId': instance_id, 'ok': True}


@app.post('/internal/sessions/raise-event')
def raise_session_event_endpoint(request: dict[str, Any]) -> dict[str, Any]:
	instance_id = str(request.get('instanceId') or '').strip()
	event_name = str(request.get('eventName') or '').strip()
	if not instance_id or not event_name:
		raise HTTPException(status_code=400, detail='instanceId + eventName required')
	raise_request = taskhub_pb.RaiseEventRequest(
		instanceId=instance_id,
		name=event_name,
		input=wrappers_pb2.StringValue(value=json.dumps(request.get('payload') or {})),
	)
	try:
		_taskhub_call('RaiseEvent', raise_request)
	except Exception as exc:
		raise HTTPException(status_code=500, detail=f'RaiseEvent failed: {exc}') from exc
	return {'ok': True}


def main() -> None:
	uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', '8002')))


if __name__ == '__main__':
	main()
