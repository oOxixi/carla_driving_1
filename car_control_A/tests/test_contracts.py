import math

import pytest

import car_control_A as contracts_package
from car_control_A.contracts import (
    CONTRACT_VERSION,
    ControlOutput,
    DrivingCommand,
    ExecutionFeedback,
    ExecutionStatus,
    LongitudinalOutput,
    LongitudinalRequest,
    RiskMetrics,
    RuntimeVehicleState,
    SignalState,
    TrafficConstraint,
)


def state() -> RuntimeVehicleState:
    return RuntimeVehicleState(
        frame=42, sim_time_s=2.1, speed_mps=8.5,
        x_m=1.0, y_m=2.0, z_m=0.2, yaw_deg=90.0, lane_id="12",
    )


def test_contract_version_and_runtime_state_uses_si_units() -> None:
    value = state()
    assert CONTRACT_VERSION == "1.0"
    assert value.speed_mps == 8.5
    assert value.frame == 42


def test_package_reexports_all_shared_contracts() -> None:
    for name in (
        "CONTRACT_VERSION", "SignalState", "RuntimeVehicleState", "DrivingCommand",
        "TrafficConstraint", "LongitudinalRequest", "ControlOutput",
        "RiskMetrics", "LongitudinalOutput", "ExecutionStatus", "ExecutionFeedback",
    ):
        assert getattr(contracts_package, name) is globals().get(name, getattr(contracts_package, name))


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), -float("inf")])
def test_all_numeric_contract_values_reject_bool_nan_and_infinity(bad: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        RuntimeVehicleState(42, bad, 1.0, 0.0, 0.0, 0.0, 0.0, "1")
    with pytest.raises((TypeError, ValueError)):
        DrivingCommand("c1", 1.0, 2.0, bad, "SET_SPEED")
    with pytest.raises((TypeError, ValueError)):
        TrafficConstraint(SignalState.GREEN, bad)
    with pytest.raises((TypeError, ValueError)):
        LongitudinalRequest(state(), 5.0, bad)
    with pytest.raises((TypeError, ValueError)):
        RiskMetrics(bad, 2.0, False)


def test_command_validates_lifetime_and_confidence() -> None:
    command = DrivingCommand("cmd-1", 1.0, 4.0, 0.9, "SET_SPEED", 10.0)
    assert command.is_ambiguous is False
    with pytest.raises(ValueError):
        DrivingCommand("cmd-1", 4.0, 1.0, 0.9, "SET_SPEED")
    with pytest.raises(ValueError):
        DrivingCommand("cmd-1", 1.0, 4.0, 1.1, "SET_SPEED")


def test_command_exposure_and_confirmation_api() -> None:
    expired = DrivingCommand("expired", 1.0, 2.0, 0.99, "STOP")
    assert expired.is_expired_at(2.0) is True
    assert expired.is_expired_at(2.01) is True

    low_confidence = DrivingCommand("low", 1.0, 2.0, 0.79, "SET_SPEED")
    ambiguous = DrivingCommand("ambiguous", 1.0, 2.0, 0.99, "SET_SPEED", is_ambiguous=True)
    explicit = DrivingCommand("explicit", 1.0, 2.0, 0.99, "SET_SPEED", confirmation_requested=True)
    assert low_confidence.requires_confirmation
    assert ambiguous.requires_confirmation
    assert explicit.requires_confirmation


def test_control_output_is_bounded_and_actuators_are_mutually_exclusive() -> None:
    assert ControlOutput(0.4, 0.0).throttle == 0.4
    with pytest.raises(ValueError):
        ControlOutput(1.01, 0.0)
    with pytest.raises(ValueError):
        ControlOutput(0.1, 0.1)


def test_longitudinal_output_carries_risk_and_control_contract() -> None:
    output = LongitudinalOutput(
        control=ControlOutput(0.0, 0.4),
        target_accel_mps2=-2.0,
        target_speed_mps=0.0,
        state="BRAKING",
        reason="stop line",
        risk=RiskMetrics(ttc_s=1.1, desired_gap_m=5.0, emergency_brake_requested=True),
    )
    assert output.control.brake == 0.4
    assert output.risk.emergency_brake_requested


@pytest.mark.parametrize("status", list(ExecutionStatus))
def test_feedback_only_exposes_declared_terminal_states(status: ExecutionStatus) -> None:
    feedback = ExecutionFeedback("cmd-1", status, 5.0, "done")
    assert feedback.status is status
    assert feedback.is_terminal


def test_optional_distances_are_allowed_but_finite_when_present() -> None:
    constraint = TrafficConstraint(SignalState.UNKNOWN, None, speed_limit_mps=None)
    assert constraint.distance_to_stop_line_m is None
    with pytest.raises(ValueError):
        TrafficConstraint(SignalState.RED, -0.1)


def test_signal_state_is_a_strict_enum_with_unknown_reserved_for_uncertainty() -> None:
    assert TrafficConstraint(SignalState.UNKNOWN, None).signal_state is SignalState.UNKNOWN
    with pytest.raises(TypeError):
        TrafficConstraint("GREEN", None)


def test_lead_measurements_are_a_coherent_pair_with_explicit_closing_semantics() -> None:
    request = LongitudinalRequest(state(), 5.0, 0.0, lead_distance_m=10.0, closing_speed_mps=2.0)
    assert request.closing_speed_mps == 2.0  # ego speed minus lead speed; positive means approaching.
    with pytest.raises(ValueError):
        LongitudinalRequest(state(), 5.0, 0.0, lead_distance_m=10.0)
    with pytest.raises(ValueError):
        LongitudinalRequest(state(), 5.0, 0.0, closing_speed_mps=2.0)


@pytest.mark.parametrize("value", [True, -0.1, float("nan"), float("inf")])
def test_speed_magnitudes_must_be_nonnegative_and_finite(value: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        RuntimeVehicleState(1, 1.0, value, 0.0, 0.0, 0.0, 0.0, "1")


@pytest.mark.parametrize(
    "value",
    [
        state(),
        DrivingCommand("command", 1.0, 2.0, 0.9, "SET_SPEED", 5.0),
        TrafficConstraint(SignalState.RED, 10.0, 8.0),
        LongitudinalRequest(state(), 5.0, 0.0, TrafficConstraint(SignalState.GREEN, None)),
        ControlOutput(0.2, 0.0),
        RiskMetrics(1.0, 3.0, False),
        LongitudinalOutput(ControlOutput(0.0, 0.4), -2.0, 0.0, "BRAKING", "stop", RiskMetrics(1.0, 3.0, True)),
        ExecutionFeedback("command", ExecutionStatus.SUCCEEDED, 2.0, "done"),
    ],
)
def test_contracts_round_trip_a_strict_versioned_dict(value: object) -> None:
    payload = value.to_dict()
    assert payload["schema_version"] == CONTRACT_VERSION
    assert type(value).from_dict(payload) == value


def test_from_dict_rejects_unknown_missing_bad_version_and_bad_types() -> None:
    payload = DrivingCommand("command", 1.0, 2.0, 0.9, "STOP").to_dict()
    with pytest.raises(ValueError):
        DrivingCommand.from_dict({**payload, "extra": 1})
    with pytest.raises(ValueError):
        DrivingCommand.from_dict({key: value for key, value in payload.items() if key != "action"})
    with pytest.raises(ValueError):
        DrivingCommand.from_dict({**payload, "schema_version": "9.9"})
    with pytest.raises(TypeError):
        DrivingCommand.from_dict({**payload, "confidence": "0.9"})


def test_nested_payload_rejects_unknown_fields_too() -> None:
    payload = LongitudinalRequest(state(), 5.0, 0.0, TrafficConstraint(SignalState.GREEN, None)).to_dict()
    payload["traffic"]["unexpected"] = "not allowed"
    with pytest.raises(ValueError):
        LongitudinalRequest.from_dict(payload)


@pytest.mark.parametrize(
    ("contract_type", "value"),
    [
        (RuntimeVehicleState, state()),
        (DrivingCommand, DrivingCommand("command", 1.0, 2.0, 0.9, "STOP")),
        (TrafficConstraint, TrafficConstraint(SignalState.GREEN, None)),
        (LongitudinalRequest, LongitudinalRequest(state(), 5.0, 0.0)),
        (ControlOutput, ControlOutput(0.0, 0.2)),
        (RiskMetrics, RiskMetrics(None, 3.0, False)),
        (LongitudinalOutput, LongitudinalOutput(ControlOutput(0.0, 0.2), -1.0, 0.0, "BRAKING", "test", RiskMetrics(None, 3.0, False))),
        (ExecutionFeedback, ExecutionFeedback("command", ExecutionStatus.SUCCEEDED, 2.0, "done")),
    ],
)
def test_each_deserializer_rejects_unknown_fields_and_schema_mismatch(contract_type: type, value: object) -> None:
    payload = value.to_dict()
    with pytest.raises(ValueError):
        contract_type.from_dict({**payload, "unknown": None})
    with pytest.raises(ValueError):
        contract_type.from_dict({**payload, "schema_version": "2.0"})
