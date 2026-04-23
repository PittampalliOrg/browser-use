"""Pydantic models for the browser-use durable runtime."""

from __future__ import annotations

from browser_use.agent import views as agent_views
from browser_use.agent.views import AgentState
from browser_use.dapr_runtime.browserstation import BrowserstationSessionState


class DurableSessionState(AgentState):
	"""Persisted runtime state for a browser-use session."""

	browserstation: BrowserstationSessionState | None = None


DurableSessionState.model_rebuild(_types_namespace=vars(agent_views))


def has_persisted_session_history(state: DurableSessionState) -> bool:
	"""Return True when a session carries state worth injecting back into the agent."""

	return state.n_steps > 1 or bool(state.message_manager_state.history.get_messages())
