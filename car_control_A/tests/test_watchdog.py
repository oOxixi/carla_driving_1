import pytest

from car_control_A.watchdog import RuntimeWatchdog


def test_watchdog_brakes_on_timeout_and_module_failure() -> None:
    watchdog = RuntimeWatchdog(timeout_s=1.0)
    watchdog.heartbeat("perception", now_s=1.0)
    assert watchdog.check(now_s=1.5) is None
    assert watchdog.check(now_s=2.1).brake == 1.0
    watchdog.heartbeat("perception", now_s=3.0)
    assert watchdog.module_failed("perception", RuntimeError("boom")).brake == 1.0
    with pytest.raises(ValueError):
        watchdog.heartbeat("", now_s=1.0)


def test_required_module_that_never_heartbeats_brakes_after_grace_and_timeout() -> None:
    watchdog = RuntimeWatchdog(timeout_s=1.0, required_modules=("perception",), startup_grace_s=0.5, started_at_s=10.0)
    assert watchdog.check(now_s=11.49) is None
    assert watchdog.check(now_s=11.5).brake == 1.0
