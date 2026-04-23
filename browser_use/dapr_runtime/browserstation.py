"""Browserstation helpers for browser-use durable sessions."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field


class BrowserstationSessionState(BaseModel):
	"""Persisted Browserstation browser metadata for one session."""

	model_config = ConfigDict(extra='forbid')

	browser_id: str
	websocket_url: str
	created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class BrowserstationClient:
	"""Minimal REST client for Browserstation."""

	def __init__(
		self,
		base_url: str,
		api_key: str | None = None,
		timeout_seconds: float = 30,
		poll_interval_seconds: float = 1,
		ready_timeout_seconds: float = 45,
	):
		self.base_url = base_url.rstrip('/')
		self.api_key = api_key
		self.timeout_seconds = timeout_seconds
		self.poll_interval_seconds = poll_interval_seconds
		self.ready_timeout_seconds = ready_timeout_seconds

	@classmethod
	def from_env(cls) -> 'BrowserstationClient':
		import os

		return cls(
			base_url=os.environ.get(
				'BROWSERSTATION_BASE_URL',
				'http://browserstation.ray-system.svc.cluster.local:8050',
			),
			api_key=os.environ.get('BROWSERSTATION_API_KEY'),
			timeout_seconds=float(os.environ.get('BROWSERSTATION_TIMEOUT_SECONDS', '30')),
			poll_interval_seconds=float(os.environ.get('BROWSERSTATION_POLL_INTERVAL_SECONDS', '1')),
			ready_timeout_seconds=float(os.environ.get('BROWSERSTATION_READY_TIMEOUT_SECONDS', '45')),
		)

	def _headers(self) -> dict[str, str]:
		headers: dict[str, str] = {}
		if self.api_key:
			headers['X-API-Key'] = self.api_key
		return headers

	async def _request(self, method: str, path: str, *, json_body: dict[str, Any] | None = None) -> dict[str, Any]:
		async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
			response = await client.request(
				method,
				f'{self.base_url}{path}',
				headers=self._headers(),
				json=json_body,
			)
			response.raise_for_status()
			data = response.json()
			return data if isinstance(data, dict) else {}

	async def create_browser(self) -> BrowserstationSessionState:
		payload = await self._request('POST', '/browsers')
		browser_id = str(payload.get('browser_id') or '').strip()
		if not browser_id:
			raise RuntimeError(f'Browserstation create_browser returned no browser_id: {payload!r}')
		return await self.wait_for_browser(browser_id)

	async def wait_for_browser(self, browser_id: str) -> BrowserstationSessionState:
		remaining = self.ready_timeout_seconds
		while remaining >= 0:
			payload = await self._request('GET', f'/browsers/{browser_id}')
			if payload.get('chrome_ready') and payload.get('websocket_url'):
				return BrowserstationSessionState(
					browser_id=browser_id,
					websocket_url=self.absolute_websocket_url(str(payload['websocket_url'])),
				)
			await asyncio.sleep(self.poll_interval_seconds)
			remaining -= self.poll_interval_seconds
		raise TimeoutError(f'Browserstation browser {browser_id} did not become ready in time')

	async def delete_browser(self, browser_id: str) -> None:
		try:
			await self._request('DELETE', f'/browsers/{browser_id}')
		except httpx.HTTPStatusError as exc:
			if exc.response.status_code == 404:
				return
			raise

	def absolute_websocket_url(self, websocket_path: str) -> str:
		"""Build an absolute websocket URL from Browserstation's relative path."""
		base = urlparse(self.base_url)
		scheme = 'wss' if base.scheme == 'https' else 'ws'
		return urljoin(f'{scheme}://{base.netloc}', websocket_path)
