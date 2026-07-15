import json

from car_control_A.telemetry import LatencyTrace


def test_trace_reports_monotonic_segments_and_writes_jsonl(tmp_path) -> None:
    trace = LatencyTrace("cmd-1")
    trace.mark("received", timestamp_ns=100)
    trace.mark("decision", timestamp_ns=150)
    trace.mark("applied", timestamp_ns=220)
    assert trace.segment_ms("received", "decision") == 0.00005
    assert trace.end_to_end_ms == 0.00012
    output = tmp_path / "trace.jsonl"
    trace.append_jsonl(output, extra={"frame": 4})
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["command_id"] == "cmd-1" and record["frame"] == 4


def test_trace_rejects_non_monotonic_and_duplicate_stage() -> None:
    trace = LatencyTrace("cmd")
    trace.mark("received", timestamp_ns=100)
    try:
        trace.mark("received", timestamp_ns=101)
    except ValueError:
        pass
    else:
        raise AssertionError("expected duplicate stage rejection")
    try:
        trace.mark("late", timestamp_ns=99)
    except ValueError:
        pass
    else:
        raise AssertionError("expected non-monotonic rejection")
