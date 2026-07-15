import math

import pytest

from car_control_A.command_adapter import CommandAdapter, CommandDisposition


def test_fast_path_parses_stop_emergency_and_chinese_speed() -> None:
    adapter = CommandAdapter(default_ttl_s=5.0)
    stop = adapter.adapt("停车", command_id="stop", now_s=1.0, confidence=0.99)
    emergency = adapter.adapt("紧急刹车", command_id="brake", now_s=1.0, confidence=0.99)
    speed = adapter.adapt("请设置到20公里每小时", command_id="speed", now_s=1.0, confidence=0.99)
    assert stop.disposition is CommandDisposition.FAST_PATH and stop.command.action == "STOP"
    assert emergency.command.action == "EMERGENCY_BRAKE"
    assert speed.command.action == "SET_SPEED"
    assert speed.command.target_speed_mps == 20 / 3.6


def test_fast_path_uses_real_utf8_chinese_and_accepts_km_h_and_whitespace() -> None:
    adapter = CommandAdapter()
    assert adapter.adapt("  停 车  ", command_id="stop", now_s=1.0, confidence=1.0).command.action == "STOP"
    assert adapter.adapt(" 紧急 刹车 ", command_id="brake", now_s=1.0, confidence=1.0).command.action == "EMERGENCY_BRAKE"
    speed = adapter.adapt("请 设置 到 20 km / h", command_id="speed", now_s=1.0, confidence=1.0)
    assert speed.command.target_speed_mps == 20 / 3.6


def test_fast_path_accepts_kmh_but_unknown_never_guesses() -> None:
    adapter = CommandAdapter()
    speed = adapter.adapt("20kmh", command_id="speed", now_s=1.0, confidence=0.99)
    unknown = adapter.adapt("从施工锥桶左侧绕开", command_id="complex", now_s=1.0, confidence=0.99)
    assert speed.command.target_speed_mps == 20 / 3.6
    assert unknown.disposition is CommandDisposition.NEEDS_DECISION
    assert unknown.command is None
    assert unknown.decision_request is not None
    assert unknown.decision_request.text == "从施工锥桶左侧绕开"


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"confidence": -0.1}, ValueError),
        ({"confidence": 1.1}, ValueError),
        ({"confidence": math.nan}, ValueError),
        ({"confidence": math.inf}, ValueError),
        ({"now_s": math.nan}, ValueError),
        ({"now_s": math.inf}, ValueError),
        ({"expires_at_s": math.nan}, ValueError),
        ({"expires_at_s": math.inf}, ValueError),
        ({"expires_at_s": 0.5}, ValueError),
    ],
)
def test_complex_command_rejects_invalid_metadata_before_decision_request(kwargs, error) -> None:
    values = {"command_id": "complex", "now_s": 1.0, "confidence": 0.9}
    values.update(kwargs)
    with pytest.raises(error):
        CommandAdapter().adapt("从施工锥桶左侧绕开", **values)


@pytest.mark.parametrize("ttl", [math.nan, math.inf])
def test_complex_command_rejects_nonfinite_default_ttl(ttl) -> None:
    with pytest.raises(ValueError):
        CommandAdapter(default_ttl_s=ttl).adapt(
            "从施工锥桶左侧绕开", command_id="complex", now_s=1.0, confidence=0.9,
        )
