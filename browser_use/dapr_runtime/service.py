"""Dapr durable runtime host for browser-use."""

from __future__ import annotations

import base64
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

from browser_use import Agent, Browser, ChatAnthropic, ChatBrowserUse, ChatGoogle, ChatOpenAI, Tools
from browser_use.agent.views import AgentHistoryList
from browser_use.dapr_runtime.browserstation import BrowserstationClient, BrowserstationSessionState
from browser_use.dapr_runtime.models import DurableSessionState, has_persisted_session_history
from browser_use.llm.base import BaseChatModel
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
	"""Persist a session event into workflow-builder, best-effort."""
	internal_token = os.environ.get('INTERNAL_API_TOKEN', '').strip()
	if not internal_token or not session_id:
		return
	app_id = os.environ.get('WORKFLOW_BUILDER_APP_ID', 'workflow-builder')
	url = (
		f"{_dapr_http_base()}/v1.0/invoke/{urllib.parse.quote(app_id, safe='')}"
		f'/method/api/internal/sessions/{urllib.parse.quote(session_id, safe="")}/events/ingest'
	)
	body = {
		'type': event_type,
		'data': data or {},
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


def _build_llm(agent_config: dict[str, Any]) -> BaseChatModel:
	model_spec = str(agent_config.get('modelSpec') or '').strip()
	if not model_spec:
		return ChatBrowserUse()

	raw_model = model_spec.split('/', 1)[1] if '/' in model_spec else model_spec
	lowered = model_spec.lower()
	if lowered.startswith('anthropic/') or raw_model.startswith('claude'):
		return ChatAnthropic(model=raw_model)
	if lowered.startswith('google/') or raw_model.startswith('gemini'):
		return ChatGoogle(model=raw_model)
	if lowered.startswith('openai/') or raw_model.startswith(('gpt', 'o1', 'o3', 'o4')):
		return ChatOpenAI(model=raw_model)
	if lowered.startswith('browser-use/') or lowered.startswith('browser_use/') or raw_model.startswith('browser-use'):
		return ChatBrowserUse(model=raw_model)
	return ChatBrowserUse(model=raw_model)


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

	async def _on_step(_browser_state: Any, model_output: Any, _step_info: Any) -> None:
		goal = getattr(getattr(model_output, 'current_state', None), 'next_goal', None)
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

	agent = Agent(
		task=task,
		llm=_build_llm(agent_config),
		browser=browser,
		tools=tools,
		extend_system_message=extend_system_message,
		llm_timeout=int(agent_config.get('timeoutMinutes') or 90) * 60,
		step_timeout=DEFAULT_STEP_TIMEOUT,
		injected_agent_state=injected_state,
		register_new_step_callback=_on_step,
	)

	history: AgentHistoryList | None = None
	result_payload: dict[str, Any] | None = None
	try:
		history = await agent.run(max_steps=max_steps)
		session_state = DurableSessionState.model_validate(agent.state.model_dump(mode='json'))
		session_state.browserstation = session_state.browserstation or _load_session_state(session_id).browserstation
		_save_session_state(session_id, session_state)
		final_content = history.final_result() or latest_step_text['value'] or ''
		if final_content:
			_publish_session_event(
				session_id,
				'agent.message',
				{
					'role': 'assistant',
					'content': [{'type': 'text', 'text': str(final_content)}],
				},
			)
		result_payload = {
			'success': bool(history.is_successful() if history.is_done() else not history.has_errors()),
			'content': str(final_content),
			'urls': history.urls(),
			'errors': [error for error in history.errors() if error],
			'sessionId': session_id,
			'turn': turn,
		}
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
	logger.info('%s starting', SERVICE_NAME)
	yield
	logger.info('%s shutting down', SERVICE_NAME)
	runner.shutdown(agent)


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
