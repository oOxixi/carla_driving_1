"""CARLA-free A/C command-to-control integration regression tests.

These tests use protocol-shaped fakes for the future B/D owners.  They prove
that A remains the command/runtime owner while C owns only longitudinal
planning; no test double implements a lateral or final-safety algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

from car_control_A import (
    ControlOutput,
    DrivingCommand,
    LongitudinalRequest,
    RuntimeVehicleState,
)
from car_control_A.behavior_fsm import BehaviorFSM, BehaviorState
from car_control_A.command_adapter import CommandAdapter, CommandDisposition
from car_control_A.routing import LateralController, RouteReference
from car_control_A.telemetry import LatencyTrace
from car_control_A.watchdog import RuntimeWatchdog
from car_control_C import FuzzyCommandPolicy, LongitudinalController


@dataclass
class FixedLateralController:
    """B-shaped test double; deliberately not a delivered lateral algorithm."""

    value: float = 0.15

    def steer(self, reference: RouteReference) -> float:
        assert reference.target_speed_mps >= 0.0
        return self.value


@dataclass
class PassThroughSafetySupervisor:
    """D-shaped test double proving A can inject an eventual final supervisor."""

    received: ControlOutput | None = None

    def arbitrate(self, control: ControlOutput) -> ControlOutput:
        self.received = control
        return control


def _vehicle(*, sim_time_s: float = 10.0, speed_mps: float = 0.0) -> RuntimeVehicleState:
    return RuntimeVehicleState(42, sim_time_s, speed_mps, 1.0, 2.0, 0.0, 0.0, "lane-1")


def _request(vehicle: RuntimeVehicleState, *, target_speed_mps: float,
             lead_distance_m: float | None = None,
             closing_speed_mps: float | None = None) -> LongitudinalRequest:
    return LongitudinalRequest(vehicle, target_speed_mps, 0.0, None,
                               lead_distance_m, closing_speed_mps)


def test_real_chinese_asr_fast_path_flows_through_fsm_c_and_injected_protocols() -> None:
    adapter = CommandAdapter(default_ttl_s=5.0)
    adapted = adapter.adapt("请设置到20公里每小时", command_id="voice-speed-1",
                            now_s=10.0, confidence=0.99)
    assert adapted.disposition is CommandDisposition.FAST_PATH
    assert adapted.command is not None
    command = adapted.command
    assert command.action == "SET_SPEED"
    assert command.target_speed_mps == 20.0 / 3.6

    trace = LatencyTrace(command.command_id)
    trace.mark("asr_received", timestamp_ns=1_000)
    fsm = BehaviorFSM()
    assert fsm.submit(command, now_s=10.0).state is BehaviorState.LANE_FOLLOW
    trace.mark("fsm_accepted", timestamp_ns=2_000)

    vehicle = _vehicle()
    output = LongitudinalController().step(
        _request(vehicle, target_speed_mps=command.target_speed_mps or 0.0), 0.05)
    trace.mark("longitudinal_planned", timestamp_ns=3_000)
    lateral: LateralController = FixedLateralController()
    steer = lateral.steer(RouteReference(((1.0, 2.0), (11.0, 2.0)), 0.0,
                                          output.target_speed_mps))
    safety = PassThroughSafetySupervisor()
    final = safety.arbitrate(ControlOutput(output.control.throttle, output.control.brake, steer))
    trace.mark("control_applied", timestamp_ns=4_000)

    feedback = fsm.complete(command.command_id, now_s=10.1, detail="speed target accepted")
    assert feedback is not None and feedback.status.value == "SUCCEEDED"
    assert final.steer == 0.15
    assert safety.received == final
    assert final.throttle > 0.0 and final.brake == 0.0
    assert trace.segment_ms("asr_received", "fsm_accepted") == 0.001
    assert trace.end_to_end_ms == 0.003


def test_low_confidence_command_enters_confirmation_and_c_safe_stop_preserves_ttc() -> None:
    command = DrivingCommand("voice-low-ttc", 10.0, 15.0, 0.1, "SET_SPEED", 12.0)
    fsm = BehaviorFSM()
    assert fsm.submit(command, now_s=10.0).state is BehaviorState.CONFIRMING

    request = _request(_vehicle(speed_mps=8.0), target_speed_mps=12.0,
                       lead_distance_m=2.0, closing_speed_mps=4.0)
    decision = FuzzyCommandPolicy().evaluate(command, request)
    assert decision.intervened and decision.requires_confirmation
    assert decision.request.requested_speed_mps == 0.0
    assert decision.output is not None
    assert decision.output.control.throttle == 0.0
    assert decision.output.risk.ttc_s == 0.5
    assert decision.output.risk.emergency_brake_requested
    assert decision.output.control.brake > 0.0

    declined = fsm.confirm(command.command_id, approved=False, now_s=10.1)
    assert declined.feedback is not None
    assert declined.feedback.status.value == "REJECTED"


def test_watchdog_failure_bypasses_normal_command_control_with_full_brake() -> None:
    watchdog = RuntimeWatchdog(timeout_s=0.5, required_modules=("command_runtime",),
                               started_at_s=0.0)
    emergency = watchdog.check(now_s=0.5)
    assert emergency == ControlOutput(0.0, 1.0, 0.0)
