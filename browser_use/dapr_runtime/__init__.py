"""Dapr durable runtime for browser-use."""

__all__ = ['app', 'main']


def __getattr__(name: str):
	if name in {'app', 'main'}:
		from browser_use.dapr_runtime.service import app, main

		return {'app': app, 'main': main}[name]
	raise AttributeError(name)
