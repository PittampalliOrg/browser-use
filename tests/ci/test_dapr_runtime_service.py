from browser_use.llm.messages import UserMessage
from browser_use.agent.views import ActionResult, AgentState
from browser_use.dapr_runtime.browserstation import BrowserstationSessionState
from browser_use.dapr_runtime import llm as runtime_llm
from browser_use.dapr_runtime.models import DurableSessionState, has_persisted_session_history


class _FakeChatModel:
	def __init__(self, model):
		self.model = model


def test_durable_session_state_validates_agent_state_dump():
	state = AgentState(last_result=[ActionResult(extracted_content='ok')])

	restored = DurableSessionState.model_validate(state.model_dump(mode='json'))

	assert restored.last_result is not None
	assert restored.last_result[0].extracted_content == 'ok'


def test_durable_session_state_round_trips_browserstation():
	state = DurableSessionState(
		last_result=[ActionResult(extracted_content='saved')],
		browserstation=BrowserstationSessionState(
			browser_id='browser-1',
			websocket_url='ws://browserstation/ws/browser-1',
		),
	)

	restored = DurableSessionState.model_validate(state.model_dump(mode='json'))

	assert restored.browserstation is not None
	assert restored.browserstation.browser_id == 'browser-1'
	assert restored.last_result is not None
	assert restored.last_result[0].extracted_content == 'saved'


def test_has_persisted_session_history_uses_message_history():
	state = DurableSessionState()
	assert has_persisted_session_history(state) is False

	state.message_manager_state.history.context_messages.append(UserMessage(content='hello'))

	assert has_persisted_session_history(state) is True


def test_build_llm_defaults_to_regular_model(monkeypatch):
	monkeypatch.setattr(runtime_llm, 'DEFAULT_MODEL_SPEC', 'openai/gpt-4.1-mini')
	monkeypatch.setattr(runtime_llm, 'ChatOpenAI', _FakeChatModel)

	llm = runtime_llm.build_llm({})

	assert isinstance(llm, _FakeChatModel)
	assert llm.model == 'gpt-4.1-mini'


def test_build_llm_rejects_browser_use_cloud_models():
	try:
		runtime_llm.build_llm({'modelSpec': 'browser-use/bu-2-0'})
	except ValueError as exc:
		assert 'Unsupported browser-use cloud model' in str(exc)
	else:
		raise AssertionError('browser-use cloud model should be rejected')


def test_build_llm_rejects_unknown_models():
	try:
		runtime_llm.build_llm({'modelSpec': 'unknown/model'})
	except ValueError as exc:
		assert 'Unsupported modelSpec' in str(exc)
	else:
		raise AssertionError('unknown model should be rejected')
