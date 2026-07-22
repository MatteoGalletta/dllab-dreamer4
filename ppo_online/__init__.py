"""Lightweight package exports for ppo_online.

Avoid eager imports here so importing submodules such as
`ppo_online.env_config` does not recursively pull in `ppo_online.networks`
and the behavioural cloning modules that depend on it.
"""

from __future__ import annotations

from importlib import import_module

__all__ = ["VectorActorCritic", "PPOVectorBuffer", "PPOAgent"]


def __getattr__(name: str):
	if name == "VectorActorCritic":
		return import_module(".networks", __name__).VectorActorCritic
	if name == "PPOVectorBuffer":
		return import_module(".buffer", __name__).PPOVectorBuffer
	if name == "PPOAgent":
		return import_module(".agent", __name__).PPOAgent
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")