"""Narrow runtime health fail-safe, deliberately not D's safety arbiter."""

from __future__ import annotations

from .contracts import ControlOutput


class RuntimeWatchdog:
    def __init__(self, *, timeout_s: float = 1.0, required_modules: tuple[str, ...] = (),
                 startup_grace_s: float = 0.0, started_at_s: float = 0.0) -> None:
        if timeout_s <= 0.0 or startup_grace_s < 0.0 or started_at_s < 0.0:
            raise ValueError("timeout_s must be positive; grace and start must be non-negative")
        if any(type(module) is not str or not module for module in required_modules):
            raise ValueError("required_modules must contain non-empty strings")
        self._timeout_s = float(timeout_s)
        self._required_modules = frozenset(required_modules)
        self._startup_deadline_s = float(started_at_s) + float(startup_grace_s) + self._timeout_s
        self._heartbeats: dict[str, float] = {}

    def heartbeat(self, module: str, *, now_s: float) -> None:
        if type(module) is not str or not module:
            raise ValueError("module must be a non-empty string")
        self._heartbeats[module] = float(now_s)

    def check(self, *, now_s: float) -> ControlOutput | None:
        if now_s >= self._startup_deadline_s and any(module not in self._heartbeats for module in self._required_modules):
            return self._full_brake()
        if any(now_s - timestamp > self._timeout_s for timestamp in self._heartbeats.values()):
            return self._full_brake()
        return None

    def module_failed(self, module: str, error: BaseException) -> ControlOutput:
        if type(module) is not str or not module:
            raise ValueError("module must be a non-empty string")
        return self._full_brake()

    @staticmethod
    def _full_brake() -> ControlOutput:
        return ControlOutput(throttle=0.0, brake=1.0, steer=0.0)
