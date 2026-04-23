from browser_use.llm.messages import UserMessage
from browser_use.agent.views import ActionResult, AgentState
from browser_use.dapr_runtime.browserstation import BrowserstationSessionState
from browser_use.dapr_runtime.models import DurableSessionState, has_persisted_session_history


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
