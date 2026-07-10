# Basic-Track Multimodal Driving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable, offline CARLA 0.9.16 closed loop that consumes ASR text plus RGB, LiDAR, ego state, and weather; aligns language to scene objects; produces safe high-level driving intents; and completes six scored scenarios on one RTX 5070 Laptop GPU with 8GB VRAM.

**Architecture:** Use one pinned external runtime dependency, ScenarioRunner v0.9.16, for scenario orchestration and criteria. Keep perception, language alignment, behavior planning, safety arbitration, route planning, and control behind typed Python interfaces. Run lightweight perception continuously, invoke a local 3B-class VLM only for complex commands, and keep deterministic safety/control active independently of model health.

**Tech Stack:** Windows 11, PowerShell, Python 3.12, CARLA 0.9.16 Python wheel, ScenarioRunner v0.9.16, NumPy, OpenCV, Ultralytics detector, PyTorch/Transformers for Qwen2.5-VL-3B-AWQ evaluation, pytest, JSON/JSONL logs.

---

## Scope decomposition and execution order

The approved specification contains four independently testable phases. Execute them in this order and do not start a phase until the previous phase's checkpoint passes:

1. **Foundation and ScenarioRunner:** Python 3.12 environment, typed contracts, CARLA synchronous client, ScenarioRunner v0.9.16, baseline safety/control.
2. **Perception and scene state:** RGB detection, LiDAR clustering, projection-based association, stable `SceneState`.
3. **Language and decision:** LMDrive format mapping, fast command parser, DriveMLM-style behavior states, target resolution, local VLM adapter.
4. **Scenarios and evaluation:** six scenario configurations, end-to-end runner, latency metrics, 20-run regression and demo freeze.

## Locked file map

```text
pyproject.toml                         # package metadata and dependency groups
.gitignore                             # generated data, models, external checkout, logs
scripts/bootstrap.ps1                  # Python 3.12 venv and CARLA wheel installation
scripts/fetch_scenario_runner.ps1      # pinned ScenarioRunner checkout
scripts/run_scenario.ps1               # CARLA + ScenarioRunner + ego agent entry point
scripts/benchmark_vlm.ps1              # local model latency/VRAM benchmark
external/scenario_runner.lock          # upstream URL, tag, commit, license
configs/runtime.json                   # ports, rates, sensor sizes, safety limits
configs/scenarios/*.json               # six scored scenario definitions
docs/references/scenario-runner.md      # adoption evidence and scenario mapping
docs/references/lmdrive-mapping.md      # LMDrive-to-local schema mapping
docs/references/drivemlm-mapping.md     # behavior-state responsibility mapping
src/carla_driving/interfaces/commands.py
src/carla_driving/interfaces/scene.py
src/carla_driving/interfaces/intent.py
src/carla_driving/simulator/client.py
src/carla_driving/simulator/sync.py
src/carla_driving/simulator/scenario_runner.py
src/carla_driving/perception/image_detector.py
src/carla_driving/perception/lidar_clusterer.py
src/carla_driving/perception/fusion.py
src/carla_driving/perception/tracker.py
src/carla_driving/alignment/fast_parser.py
src/carla_driving/alignment/target_resolver.py
src/carla_driving/alignment/vlm_adapter.py
src/carla_driving/decision/orchestrator.py
src/carla_driving/decision/sequencer.py
src/carla_driving/safety/arbiter.py
src/carla_driving/planning/route_planner.py
src/carla_driving/control/ego_agent.py
src/carla_driving/evaluation/latency.py
src/carla_driving/evaluation/result.py
src/carla_driving/cli/run_agent.py
tests/unit/...                         # pure unit tests mirroring src packages
tests/integration/...                  # CARLA-free integration tests with fakes
tests/smoke/...                        # requires running CARLA/ScenarioRunner
```

Do not commit CARLA binaries, model weights, recorded sensor data, or the ScenarioRunner checkout. Commit only the lock record, installation scripts, source, tests, configs, and small deterministic fixtures.

## Phase 1: Foundation and ScenarioRunner

### Task 1: Establish the Python 3.12 project and reproducible environment

**Files:**
- Create: `pyproject.toml`
- Create: `scripts/bootstrap.ps1`
- Modify: `.gitignore`
- Create: `src/carla_driving/__init__.py`
- Create: `tests/unit/test_environment_contract.py`

- [ ] **Step 1: Write the environment contract test**

```python
# tests/unit/test_environment_contract.py
import sys


def test_supported_python_runtime():
    assert sys.version_info[:2] == (3, 12)


def test_package_imports():
    import carla_driving

    assert carla_driving.__version__ == "0.1.0"
```

- [ ] **Step 2: Run the test with the current interpreter and record the expected failure**

Run: `python -m pytest tests/unit/test_environment_contract.py -v`

Expected: collection fails because pytest/package setup is absent, and the current default Python is 3.13 rather than 3.12.

- [ ] **Step 3: Create package metadata and bootstrap script**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "carla-driving-rstar"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "numpy>=2.0,<3",
  "pygame>=2.6,<3",
]

[project.optional-dependencies]
dev = ["pytest>=8,<9", "pytest-cov>=6,<7"]
perception = ["opencv-python>=4.10,<5", "ultralytics>=8.3,<9"]
vlm = [
  "torch>=2.6,<3",
  "transformers>=4.51,<5",
  "accelerate>=1.5,<2",
  "qwen-vl-utils>=0.0.10,<1",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
addopts = "-ra"
```

```powershell
# scripts/bootstrap.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = "py"

& $Python -3.12 -m venv "$Root\.venv"
$VenvPython = "$Root\.venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -e "$Root[dev]"
& $VenvPython -m pip install "$Root\CARLA_0.9.16\PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl"
& $VenvPython -c "import carla; print('CARLA Python API import passed')"
& $VenvPython -m pytest "$Root\tests\unit\test_environment_contract.py" -v
```

```python
# src/carla_driving/__init__.py
__version__ = "0.1.0"
```

Append these exact entries to `.gitignore`:

```gitignore
.venv/
.pytest_cache/
.coverage
htmlcov/
external/scenario_runner/
models/
artifacts/
```

- [ ] **Step 4: Bootstrap and verify**

Run: `powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1`

Expected: `CARLA Python API import passed` and `2 passed`.

- [ ] **Step 5: Commit only Task 1 files**

```powershell
git add pyproject.toml .gitignore scripts/bootstrap.ps1 src/carla_driving/__init__.py tests/unit/test_environment_contract.py
git commit -m "build: add Python 3.12 CARLA development environment"
```

### Task 2: Define JSON-safe public contracts

**Files:**
- Create: `src/carla_driving/interfaces/__init__.py`
- Create: `src/carla_driving/interfaces/commands.py`
- Create: `src/carla_driving/interfaces/scene.py`
- Create: `src/carla_driving/interfaces/intent.py`
- Create: `tests/unit/interfaces/test_contracts.py`

- [ ] **Step 1: Write failing contract tests**

```python
# tests/unit/interfaces/test_contracts.py
import pytest

from carla_driving.interfaces.commands import ASRCommand
from carla_driving.interfaces.intent import Action, ActionStep, DrivingIntent
from carla_driving.interfaces.scene import EgoState, SceneObject, SceneState, WeatherState


def test_asr_command_round_trip():
    command = ASRCommand.from_dict({
        "command_id": "cmd-1",
        "text": "绕开前面的行人后加速到20公里每小时",
        "confidence": 0.97,
        "timestamp_ms": 123456789,
    })
    assert command.to_dict()["text"].startswith("绕开")


def test_asr_command_rejects_bad_confidence():
    with pytest.raises(ValueError, match="confidence"):
        ASRCommand("cmd-1", "停车", 1.2, 1)


def test_scene_and_intent_reference_same_object_id():
    scene = SceneState(
        frame_id=10,
        timestamp_ms=1000,
        ego=EgoState(speed_kmh=12.0, lane_id=1, steering=0.0),
        weather=WeatherState(rain=0.0, night=False),
        objects=(SceneObject("ped-1", "pedestrian", "front", 9.0, -1.0, 0.95, "high"),),
        can_change_left=True,
        can_change_right=False,
        time_to_collision_s=2.4,
    )
    intent = DrivingIntent(
        command_id="cmd-1",
        actions=(ActionStep(Action.AVOID_OBJECT, target_id="ped-1"),),
        confidence=0.94,
        reason="front pedestrian",
    )
    assert intent.actions[0].target_id == scene.objects[0].object_id
```

- [ ] **Step 2: Run the tests and verify the import failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/interfaces/test_contracts.py -v`

Expected: FAIL because the interface modules do not exist.

- [ ] **Step 3: Implement the command contract**

```python
# src/carla_driving/interfaces/commands.py
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ASRCommand:
    command_id: str
    text: str
    confidence: float
    timestamp_ms: int

    def __post_init__(self) -> None:
        if not self.command_id.strip():
            raise ValueError("command_id must not be empty")
        if not self.text.strip():
            raise ValueError("text must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")

    @classmethod
    def from_dict(cls, value: dict) -> "ASRCommand":
        return cls(
            command_id=str(value["command_id"]),
            text=str(value["text"]),
            confidence=float(value["confidence"]),
            timestamp_ms=int(value["timestamp_ms"]),
        )

    def to_dict(self) -> dict:
        return asdict(self)
```

- [ ] **Step 4: Implement scene and intent contracts**

```python
# src/carla_driving/interfaces/scene.py
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class EgoState:
    speed_kmh: float
    lane_id: int
    steering: float


@dataclass(frozen=True, slots=True)
class WeatherState:
    rain: float
    night: bool


@dataclass(frozen=True, slots=True)
class SceneObject:
    object_id: str
    object_type: str
    direction: str
    distance_m: float
    relative_speed_mps: float
    confidence: float
    risk: str


@dataclass(frozen=True, slots=True)
class SceneState:
    frame_id: int
    timestamp_ms: int
    ego: EgoState
    weather: WeatherState
    objects: tuple[SceneObject, ...]
    can_change_left: bool
    can_change_right: bool
    time_to_collision_s: float | None
    keyframe_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)
```

```python
# src/carla_driving/interfaces/intent.py
from dataclasses import asdict, dataclass
from enum import StrEnum


class Action(StrEnum):
    START = "START"
    STOP = "STOP"
    SET_SPEED = "SET_SPEED"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    CHANGE_LANE_LEFT = "CHANGE_LANE_LEFT"
    CHANGE_LANE_RIGHT = "CHANGE_LANE_RIGHT"
    AVOID_OBJECT = "AVOID_OBJECT"
    EMERGENCY_BRAKE = "EMERGENCY_BRAKE"
    RETURN_TO_LANE = "RETURN_TO_LANE"


@dataclass(frozen=True, slots=True)
class ActionStep:
    action: Action
    target_id: str | None = None
    target_speed_kmh: float | None = None


@dataclass(frozen=True, slots=True)
class DrivingIntent:
    command_id: str
    actions: tuple[ActionStep, ...]
    confidence: float
    reason: str

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("actions must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    def to_dict(self) -> dict:
        return asdict(self)
```

Create empty package exports:

```python
# src/carla_driving/interfaces/__init__.py
from .commands import ASRCommand
from .intent import Action, ActionStep, DrivingIntent
from .scene import EgoState, SceneObject, SceneState, WeatherState

__all__ = [
    "ASRCommand", "Action", "ActionStep", "DrivingIntent",
    "EgoState", "SceneObject", "SceneState", "WeatherState",
]
```

- [ ] **Step 5: Run and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/interfaces/test_contracts.py -v`

Expected: `3 passed`.

```powershell
git add src/carla_driving/interfaces tests/unit/interfaces
git commit -m "feat: define multimodal driving contracts"
```

### Task 3: Pin ScenarioRunner and document all three open-source studies

**Files:**
- Create: `scripts/fetch_scenario_runner.ps1`
- Create: `external/scenario_runner.lock`
- Create: `docs/references/scenario-runner.md`
- Create: `docs/references/lmdrive-mapping.md`
- Create: `docs/references/drivemlm-mapping.md`
- Create: `tests/unit/test_reference_contracts.py`

- [ ] **Step 1: Write a test that enforces the promised research artifacts**

```python
# tests/unit/test_reference_contracts.py
from pathlib import Path


def test_required_reference_records_exist():
    root = Path(__file__).parents[2]
    expected = {
        "external/scenario_runner.lock": ("v0.9.16", "MIT"),
        "docs/references/scenario-runner.md": ("PedestrianCrossing", "StaticCutIn"),
        "docs/references/lmdrive-mapping.md": ("ASRCommand", "SceneState"),
        "docs/references/drivemlm-mapping.md": ("DrivingIntent", "VehicleControl"),
    }
    for relative_path, terms in expected.items():
        text = (root / relative_path).read_text(encoding="utf-8")
        assert all(term in text for term in terms)
```

- [ ] **Step 2: Run the test and verify it fails on missing files**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_reference_contracts.py -v`

Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 3: Create the pinned checkout script and lock record**

```powershell
# scripts/fetch_scenario_runner.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Target = "$Root\external\scenario_runner"
$Url = "https://github.com/carla-simulator/scenario_runner.git"
$Tag = "v0.9.16"

if (Test-Path $Target) {
    $Existing = (& git -C $Target remote get-url origin).Trim()
    if ($Existing -ne $Url) { throw "Unexpected ScenarioRunner origin: $Existing" }
    & git -C $Target fetch --tags --force
} else {
    & git clone --branch $Tag --depth 1 $Url $Target
}
& git -C $Target checkout --detach $Tag
$Commit = (& git -C $Target rev-parse HEAD).Trim()
Write-Output "ScenarioRunner $Tag pinned at $Commit"
```

```text
# external/scenario_runner.lock
url=https://github.com/carla-simulator/scenario_runner.git
tag=v0.9.16
commit=94ff3b8af752bad2b9d464ad5105868906aa34c0
license=MIT
```

After checkout, assert that `git -C external/scenario_runner rev-parse HEAD` equals `94ff3b8af752bad2b9d464ad5105868906aa34c0`; abort the task on any mismatch.

- [ ] **Step 4: Write the three bounded research records**

`docs/references/scenario-runner.md` must contain a table mapping `PedestrianCrossing`, `ConstructionObstacleTwoWays`, and `StaticCutIn` to the three scored scenario families, plus exact Windows smoke commands and output artifact paths.

`docs/references/lmdrive-mapping.md` must map LMDrive navigation instruction, notice instruction, RGB/LiDAR clip, measurement, and control fields to `ASRCommand`, `SceneState`, `DrivingIntent`, and non-runtime training labels. It must explicitly exclude CARLA 0.9.10.1, LAVIS, and the 7B runtime.

`docs/references/drivemlm-mapping.md` must map DriveMLM behavior planning states to `DrivingIntent.actions`, state that continuous `VehicleControl` belongs to the planner/controller, and state that safety arbitration can override any model result.

- [ ] **Step 5: Verify the records and commit**

Run: `powershell -ExecutionPolicy Bypass -File .\scripts\fetch_scenario_runner.ps1`

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_reference_contracts.py -v`

Expected: ScenarioRunner prints the pinned commit and the test reports `1 passed`.

```powershell
git add scripts/fetch_scenario_runner.ps1 external/scenario_runner.lock docs/references tests/unit/test_reference_contracts.py
git commit -m "docs: pin ScenarioRunner and record driving model mappings"
```

### Task 4: Add a synchronous CARLA client with frame-aligned sensor buffering

**Files:**
- Create: `src/carla_driving/simulator/__init__.py`
- Create: `src/carla_driving/simulator/client.py`
- Create: `src/carla_driving/simulator/sync.py`
- Create: `tests/unit/simulator/test_sync.py`
- Create: `tests/smoke/test_carla_connection.py`

- [ ] **Step 1: Write CARLA-free tests using fake world settings and sensor frames**

```python
# tests/unit/simulator/test_sync.py
from dataclasses import dataclass

from carla_driving.simulator.client import SynchronousWorld
from carla_driving.simulator.sync import FrameBuffer


@dataclass
class Settings:
    synchronous_mode: bool = False
    fixed_delta_seconds: float | None = None


class FakeWorld:
    def __init__(self):
        self.settings = Settings()
        self.applied = []

    def get_settings(self):
        return self.settings

    def apply_settings(self, settings):
        self.settings = settings
        self.applied.append((settings.synchronous_mode, settings.fixed_delta_seconds))


def test_synchronous_world_restores_settings():
    world = FakeWorld()
    with SynchronousWorld(world, fixed_delta_seconds=0.05):
        assert world.settings.synchronous_mode is True
    assert world.applied[-1] == (False, None)


def test_frame_buffer_returns_only_matching_frame():
    buffer = FrameBuffer(max_frames=3)
    buffer.push(9, "old")
    buffer.push(10, "rgb")
    assert buffer.pop(10) == "rgb"
    assert buffer.pop(9) == "old"
```

- [ ] **Step 2: Run and verify missing-module failures**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/simulator/test_sync.py -v`

Expected: FAIL because simulator modules do not exist.

- [ ] **Step 3: Implement settings restoration and bounded frame storage**

```python
# src/carla_driving/simulator/client.py
from copy import copy


class SynchronousWorld:
    def __init__(self, world, fixed_delta_seconds: float = 0.05):
        self.world = world
        self.fixed_delta_seconds = fixed_delta_seconds
        self._original = None

    def __enter__(self):
        self._original = copy(self.world.get_settings())
        active = copy(self._original)
        active.synchronous_mode = True
        active.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(active)
        return self.world

    def __exit__(self, exc_type, exc, traceback):
        self.world.apply_settings(self._original)
```

```python
# src/carla_driving/simulator/sync.py
from collections import OrderedDict
from threading import Condition


class FrameBuffer:
    def __init__(self, max_frames: int = 8):
        self.max_frames = max_frames
        self._frames = OrderedDict()
        self._condition = Condition()

    def push(self, frame_id: int, value) -> None:
        with self._condition:
            self._frames[frame_id] = value
            while len(self._frames) > self.max_frames:
                self._frames.popitem(last=False)
            self._condition.notify_all()

    def pop(self, frame_id: int, timeout_s: float = 0.5):
        with self._condition:
            ready = self._condition.wait_for(lambda: frame_id in self._frames, timeout_s)
            if not ready:
                raise TimeoutError(f"frame {frame_id} did not arrive within {timeout_s}s")
            return self._frames.pop(frame_id)
```

- [ ] **Step 4: Add a real connection smoke test gated by an environment variable**

```python
# tests/smoke/test_carla_connection.py
import os
import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("CARLA_SMOKE") != "1",
    reason="set CARLA_SMOKE=1 with a running CARLA server",
)


def test_world_ticks_in_synchronous_mode():
    import carla
    from carla_driving.simulator.client import SynchronousWorld

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    with SynchronousWorld(world, 0.05):
        assert world.tick() > 0
```

- [ ] **Step 5: Run unit tests, then smoke test, then commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/simulator/test_sync.py -v`

Expected: `2 passed`.

With CARLA running, run:

```powershell
$env:CARLA_SMOKE="1"
.\.venv\Scripts\python.exe -m pytest tests/smoke/test_carla_connection.py -v
```

Expected: `1 passed`.

```powershell
git add src/carla_driving/simulator tests/unit/simulator tests/smoke/test_carla_connection.py
git commit -m "feat: add synchronous CARLA frame handling"
```

### Task 5: Wrap ScenarioRunner invocation and prove ego-agent handoff

**Files:**
- Create: `src/carla_driving/simulator/scenario_runner.py`
- Create: `scripts/run_scenario.ps1`
- Create: `tests/unit/simulator/test_scenario_runner.py`
- Update: `docs/references/scenario-runner.md`

- [ ] **Step 1: Test exact ScenarioRunner command construction**

```python
# tests/unit/simulator/test_scenario_runner.py
from pathlib import Path

from carla_driving.simulator.scenario_runner import build_scenario_command


def test_build_pedestrian_smoke_command():
    command = build_scenario_command(
        python=Path(".venv/Scripts/python.exe"),
        scenario_root=Path("external/scenario_runner"),
        scenario="PedestrianCrossing",
        host="127.0.0.1",
        port=2000,
    )
    assert command[-6:] == [
        "--scenario", "PedestrianCrossing", "--host", "127.0.0.1", "--port", "2000"
    ]
    assert "--sync" in command
```

- [ ] **Step 2: Run and verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/simulator/test_scenario_runner.py -v`

Expected: FAIL because `build_scenario_command` does not exist.

- [ ] **Step 3: Implement the command builder**

```python
# src/carla_driving/simulator/scenario_runner.py
from pathlib import Path


def build_scenario_command(
    python: Path,
    scenario_root: Path,
    scenario: str,
    host: str,
    port: int,
) -> list[str]:
    return [
        str(python),
        str(scenario_root / "scenario_runner.py"),
        "--sync",
        "--reloadWorld",
        "--output",
        "--scenario", scenario,
        "--host", host,
        "--port", str(port),
    ]
```

```powershell
# scripts/run_scenario.ps1
param(
    [string]$Scenario = "PedestrianCrossing",
    [int]$Port = 2000
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = "$Root\CARLA_0.9.16\PythonAPI;$Root\external\scenario_runner;$Root\src"
& "$Root\.venv\Scripts\python.exe" `
  "$Root\external\scenario_runner\scenario_runner.py" `
  --sync --reloadWorld --output --scenario $Scenario --host 127.0.0.1 --port $Port
```

- [ ] **Step 4: Run the official pedestrian smoke scenario and record evidence**

Run: `powershell -ExecutionPolicy Bypass -File .\scripts\run_scenario.ps1 -Scenario PedestrianCrossing`

Expected: ScenarioRunner connects to CARLA, spawns the pedestrian scenario, prints criteria, and exits without an import/version error. Record command, commit hash, runtime, and result path in `docs/references/scenario-runner.md`.

- [ ] **Step 5: Run tests and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/simulator/test_scenario_runner.py -v`

Expected: `1 passed`.

```powershell
git add src/carla_driving/simulator/scenario_runner.py scripts/run_scenario.ps1 tests/unit/simulator/test_scenario_runner.py docs/references/scenario-runner.md
git commit -m "feat: integrate ScenarioRunner 0.9.16 smoke scenarios"
```

### Task 6: Implement deterministic safety arbitration before vehicle control

**Files:**
- Create: `src/carla_driving/safety/__init__.py`
- Create: `src/carla_driving/safety/arbiter.py`
- Create: `tests/unit/safety/test_arbiter.py`

- [ ] **Step 1: Write safety priority tests**

```python
# tests/unit/safety/test_arbiter.py
from carla_driving.interfaces.intent import Action, ActionStep, DrivingIntent
from carla_driving.interfaces.scene import EgoState, SceneState, WeatherState
from carla_driving.safety.arbiter import SafetyArbiter


def scene(ttc, left=True, right=True):
    return SceneState(1, 1, EgoState(30.0, 1, 0.0), WeatherState(0.0, False), (), left, right, ttc)


def intent(action):
    return DrivingIntent("c1", (ActionStep(action),), 0.9, "test")


def test_emergency_brake_overrides_change_lane():
    result = SafetyArbiter(emergency_ttc_s=1.5).arbitrate(
        intent(Action.CHANGE_LANE_LEFT), scene(1.0)
    )
    assert result.actions[0].action is Action.EMERGENCY_BRAKE


def test_blocked_lane_becomes_stop():
    result = SafetyArbiter(emergency_ttc_s=1.5).arbitrate(
        intent(Action.CHANGE_LANE_LEFT), scene(None, left=False)
    )
    assert result.actions[0].action is Action.STOP
```

- [ ] **Step 2: Run and verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/safety/test_arbiter.py -v`

Expected: FAIL because the arbiter does not exist.

- [ ] **Step 3: Implement minimum-risk overrides**

```python
# src/carla_driving/safety/arbiter.py
from carla_driving.interfaces.intent import Action, ActionStep, DrivingIntent
from carla_driving.interfaces.scene import SceneState


class SafetyArbiter:
    def __init__(self, emergency_ttc_s: float = 1.5):
        self.emergency_ttc_s = emergency_ttc_s

    def arbitrate(self, intent: DrivingIntent, scene: SceneState) -> DrivingIntent:
        if scene.time_to_collision_s is not None and scene.time_to_collision_s <= self.emergency_ttc_s:
            return DrivingIntent(
                intent.command_id,
                (ActionStep(Action.EMERGENCY_BRAKE),),
                1.0,
                "safety override: collision time below threshold",
            )
        first = intent.actions[0].action
        blocked = (
            first is Action.CHANGE_LANE_LEFT and not scene.can_change_left
        ) or (
            first is Action.CHANGE_LANE_RIGHT and not scene.can_change_right
        )
        if blocked:
            return DrivingIntent(
                intent.command_id,
                (ActionStep(Action.STOP),),
                1.0,
                "safety override: target lane unavailable",
            )
        return intent
```

- [ ] **Step 4: Run and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/safety/test_arbiter.py -v`

Expected: `2 passed`.

```powershell
git add src/carla_driving/safety tests/unit/safety
git commit -m "feat: add deterministic driving safety arbiter"
```

**Phase 1 checkpoint:** CARLA imports under Python 3.12; ScenarioRunner v0.9.16 runs one official scenario; typed contracts round-trip; safety tests pass; no model package has been installed into the CARLA baseline environment.

## Phase 2: Perception and scene state

### Task 7: Implement camera detector and LiDAR clustering adapters

**Files:**
- Create: `src/carla_driving/perception/__init__.py`
- Create: `src/carla_driving/perception/image_detector.py`
- Create: `src/carla_driving/perception/lidar_clusterer.py`
- Create: `tests/unit/perception/test_lidar_clusterer.py`
- Create: `tests/unit/perception/test_image_detector.py`

- [ ] **Step 1: Test detector normalization and separated point clusters**

```python
# tests/unit/perception/test_lidar_clusterer.py
import numpy as np
from carla_driving.perception.lidar_clusterer import EuclideanClusterer


def test_two_separated_clusters():
    points = np.array([
        [5.0, 0.0, 0.5], [5.2, 0.1, 0.5], [4.9, -0.1, 0.6],
        [12.0, 3.0, 0.5], [12.2, 3.1, 0.6], [11.9, 2.9, 0.5],
    ], dtype=np.float32)
    clusters = EuclideanClusterer(radius_m=0.5, min_points=3).cluster(points)
    assert len(clusters) == 2
    assert clusters[0].distance_m < clusters[1].distance_m
```

```python
# tests/unit/perception/test_image_detector.py
from carla_driving.perception.image_detector import normalize_class_name


def test_class_normalization():
    assert normalize_class_name("person") == "pedestrian"
    assert normalize_class_name("car") == "vehicle"
    assert normalize_class_name("traffic cone") == "cone"
```

- [ ] **Step 2: Run and verify failures**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/perception -v`

Expected: FAIL because perception modules do not exist.

- [ ] **Step 3: Implement the pure NumPy clusterer and detector protocol**

```python
# src/carla_driving/perception/lidar_clusterer.py
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class PointCluster:
    cluster_id: str
    centroid: tuple[float, float, float]
    distance_m: float
    points: np.ndarray


class EuclideanClusterer:
    def __init__(self, radius_m: float = 0.8, min_points: int = 4):
        self.radius_m = radius_m
        self.min_points = min_points

    def cluster(self, points: np.ndarray) -> tuple[PointCluster, ...]:
        points = points[(points[:, 0] > 0.0) & (points[:, 0] < 50.0) & (abs(points[:, 1]) < 15.0)]
        unused = set(range(len(points)))
        groups = []
        while unused:
            seed = unused.pop()
            group = {seed}
            frontier = [seed]
            while frontier:
                current = frontier.pop()
                candidates = list(unused)
                if not candidates:
                    continue
                distances = np.linalg.norm(points[candidates, :2] - points[current, :2], axis=1)
                neighbours = [candidates[i] for i, distance in enumerate(distances) if distance <= self.radius_m]
                for neighbour in neighbours:
                    unused.remove(neighbour)
                    group.add(neighbour)
                    frontier.append(neighbour)
            if len(group) >= self.min_points:
                groups.append(points[sorted(group)])
        result = []
        for index, group in enumerate(groups):
            centroid = group.mean(axis=0)
            result.append(PointCluster(
                f"lidar-{index}", tuple(float(v) for v in centroid),
                float(np.linalg.norm(centroid[:2])), group,
            ))
        return tuple(sorted(result, key=lambda item: item.distance_m))
```

```python
# src/carla_driving/perception/image_detector.py
from dataclasses import dataclass
from typing import Protocol
import numpy as np


CLASS_MAP = {"person": "pedestrian", "car": "vehicle", "truck": "vehicle", "bus": "vehicle", "traffic cone": "cone"}


def normalize_class_name(name: str) -> str:
    return CLASS_MAP.get(name.lower(), name.lower().replace(" ", "_"))


@dataclass(frozen=True, slots=True)
class Detection2D:
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


class ImageDetector(Protocol):
    def detect(self, image_bgr: np.ndarray) -> tuple[Detection2D, ...]: ...


class UltralyticsDetector:
    def __init__(self, weights: str, confidence: float = 0.35):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.confidence = confidence

    def detect(self, image_bgr: np.ndarray) -> tuple[Detection2D, ...]:
        result = self.model.predict(image_bgr, conf=self.confidence, verbose=False)[0]
        names = result.names
        return tuple(
            Detection2D(normalize_class_name(names[int(cls)]), float(conf), tuple(float(v) for v in box))
            for box, conf, cls in zip(result.boxes.xyxy.cpu(), result.boxes.conf.cpu(), result.boxes.cls.cpu())
        )
```

- [ ] **Step 4: Install only the perception group, run tests, and commit**

Run: `.\.venv\Scripts\python.exe -m pip install -e ".[perception]"`

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/perception -v`

Expected: `2 passed`.

```powershell
git add src/carla_driving/perception tests/unit/perception
git commit -m "feat: add camera and lidar perception adapters"
```

### Task 8: Fuse detections with point depth and construct SceneState

**Files:**
- Create: `src/carla_driving/perception/fusion.py`
- Create: `src/carla_driving/perception/tracker.py`
- Create: `tests/unit/perception/test_fusion.py`

- [ ] **Step 1: Test a projected LiDAR point inside a pedestrian box**

```python
# tests/unit/perception/test_fusion.py
import numpy as np
from carla_driving.perception.fusion import associate_depth
from carla_driving.perception.image_detector import Detection2D


def test_box_receives_median_projected_depth():
    detections = (Detection2D("pedestrian", 0.9, (100, 100, 200, 300)),)
    pixels = np.array([[150, 150, 9.0], [160, 200, 11.0], [400, 100, 4.0]], dtype=np.float32)
    fused = associate_depth(detections, pixels)
    assert fused[0].distance_m == 10.0
    assert fused[0].direction == "front"


def test_object_id_stays_stable_across_nearby_frames():
    from carla_driving.perception.tracker import ObjectTracker

    tracker = ObjectTracker(max_distance_delta_m=2.0)
    first = tracker.update((associate_depth(
        (Detection2D("pedestrian", 0.9, (100, 100, 200, 300)),),
        np.array([[150, 150, 10.0]], dtype=np.float32),
    )[0],))
    second = tracker.update((associate_depth(
        (Detection2D("pedestrian", 0.92, (102, 100, 202, 300)),),
        np.array([[152, 150, 9.2]], dtype=np.float32),
    )[0],))
    assert first[0].object_id == second[0].object_id
```

- [ ] **Step 2: Run and verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/perception/test_fusion.py -v`

Expected: FAIL because fusion is absent.

- [ ] **Step 3: Implement deterministic association output**

```python
# src/carla_driving/perception/fusion.py
from dataclasses import dataclass
import numpy as np
from .image_detector import Detection2D


@dataclass(frozen=True, slots=True)
class FusedObject:
    object_id: str
    object_type: str
    confidence: float
    distance_m: float
    direction: str


def associate_depth(detections: tuple[Detection2D, ...], projected_xyz: np.ndarray) -> tuple[FusedObject, ...]:
    result = []
    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = detection.xyxy
        inside = projected_xyz[
            (projected_xyz[:, 0] >= x1) & (projected_xyz[:, 0] <= x2)
            & (projected_xyz[:, 1] >= y1) & (projected_xyz[:, 1] <= y2)
            & (projected_xyz[:, 2] > 0)
        ]
        if not len(inside):
            continue
        center_x = (x1 + x2) / 2
        direction = "left" if center_x < 213 else "right" if center_x > 427 else "front"
        result.append(FusedObject(
            object_id=f"{detection.class_name}-{index}",
            object_type=detection.class_name,
            confidence=detection.confidence,
            distance_m=float(np.median(inside[:, 2])),
            direction=direction,
        ))
    return tuple(result)
```

```python
# src/carla_driving/perception/tracker.py
from dataclasses import replace
from .fusion import FusedObject


class ObjectTracker:
    def __init__(self, max_distance_delta_m: float = 2.0):
        self.max_distance_delta_m = max_distance_delta_m
        self._previous: tuple[FusedObject, ...] = ()
        self._next_id = 1

    def update(self, current: tuple[FusedObject, ...]) -> tuple[FusedObject, ...]:
        assigned = []
        unused_previous = list(self._previous)
        for item in current:
            matches = [
                previous for previous in unused_previous
                if previous.object_type == item.object_type
                and previous.direction == item.direction
                and abs(previous.distance_m - item.distance_m) <= self.max_distance_delta_m
            ]
            if matches:
                match = min(matches, key=lambda previous: abs(previous.distance_m - item.distance_m))
                unused_previous.remove(match)
                assigned.append(replace(item, object_id=match.object_id))
            else:
                object_id = f"{item.object_type}-{self._next_id}"
                self._next_id += 1
                assigned.append(replace(item, object_id=object_id))
        self._previous = tuple(assigned)
        return self._previous
```

- [ ] **Step 4: Add integration assertions when building SceneState**

Extend `tests/unit/perception/test_fusion.py` to convert `FusedObject` instances into `SceneObject` instances and assert that frame ID, sensor timestamp, direction, distance, and risk survive serialization through `SceneState.to_dict()`.

- [ ] **Step 5: Run and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/perception tests/unit/interfaces -v`

Expected: all perception and interface tests pass.

```powershell
git add src/carla_driving/perception/fusion.py src/carla_driving/perception/tracker.py tests/unit/perception/test_fusion.py
git commit -m "feat: fuse image detections with lidar depth"
```

**Phase 2 checkpoint:** A recorded RGB/LiDAR frame produces a serialized `SceneState` containing at least one correctly typed, directed, and ranged pedestrian/vehicle/obstacle without reading CARLA actor truth at runtime.

## Phase 3: Language alignment and decision

### Task 9: Implement the fast Chinese command parser and target resolver

**Files:**
- Create: `src/carla_driving/alignment/__init__.py`
- Create: `src/carla_driving/alignment/fast_parser.py`
- Create: `src/carla_driving/alignment/target_resolver.py`
- Create: `tests/unit/alignment/test_fast_parser.py`
- Create: `tests/unit/alignment/test_target_resolver.py`

- [ ] **Step 1: Test basic and compound commands**

```python
# tests/unit/alignment/test_fast_parser.py
from carla_driving.alignment.fast_parser import parse_fast
from carla_driving.interfaces.intent import Action


def test_parse_stop():
    assert parse_fast("前方危险，减速停车")[0].action is Action.STOP


def test_parse_avoid_then_speed():
    steps = parse_fast("绕开前面的行人后加速到20公里每小时")
    assert [step.action for step in steps] == [Action.AVOID_OBJECT, Action.SET_SPEED]
    assert steps[1].target_speed_kmh == 20
```

```python
# tests/unit/alignment/test_target_resolver.py
from carla_driving.alignment.target_resolver import resolve_target
from carla_driving.interfaces.scene import SceneObject


def test_nearest_front_pedestrian_wins():
    objects = (
        SceneObject("ped-far", "pedestrian", "front", 20.0, 0.0, 0.9, "medium"),
        SceneObject("ped-near", "pedestrian", "front", 8.0, -1.0, 0.95, "high"),
        SceneObject("car-near", "vehicle", "front", 5.0, 0.0, 0.99, "high"),
    )
    assert resolve_target(objects, "pedestrian", "front").object_id == "ped-near"
```

- [ ] **Step 2: Run and verify failures**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/alignment -v`

Expected: FAIL because alignment modules do not exist.

- [ ] **Step 3: Implement bounded parsing and deterministic resolution**

```python
# src/carla_driving/alignment/fast_parser.py
import re
from carla_driving.interfaces.intent import Action, ActionStep


SPEED = re.compile(r"(?:到|至)?\s*(\d+(?:\.\d+)?)\s*(?:公里|千米)(?:每小时|/小时)?")


def parse_fast(text: str) -> tuple[ActionStep, ...]:
    if "绕开" in text or "避让" in text:
        steps = [ActionStep(Action.AVOID_OBJECT)]
        match = SPEED.search(text)
        if match:
            steps.append(ActionStep(Action.SET_SPEED, target_speed_kmh=float(match.group(1))))
        return tuple(steps)
    if "紧急" in text and ("刹" in text or "停车" in text):
        return (ActionStep(Action.EMERGENCY_BRAKE),)
    if "停车" in text or "停止" in text:
        return (ActionStep(Action.STOP),)
    if "向左变道" in text or "变到左" in text:
        return (ActionStep(Action.CHANGE_LANE_LEFT),)
    if "向右变道" in text or "变到右" in text:
        return (ActionStep(Action.CHANGE_LANE_RIGHT),)
    match = SPEED.search(text)
    if match:
        return (ActionStep(Action.SET_SPEED, target_speed_kmh=float(match.group(1))),)
    raise ValueError("command requires multimodal decision path")
```

```python
# src/carla_driving/alignment/target_resolver.py
from carla_driving.interfaces.scene import SceneObject


def resolve_target(
    objects: tuple[SceneObject, ...],
    object_type: str,
    direction: str,
) -> SceneObject:
    candidates = [
        item for item in objects
        if item.object_type == object_type and item.direction == direction and item.confidence >= 0.5
    ]
    if not candidates:
        raise LookupError(f"no {direction} {object_type} candidate")
    return min(candidates, key=lambda item: (item.distance_m, -item.confidence))
```

- [ ] **Step 4: Bind the resolved target ID into AVOID_OBJECT and test ambiguity failure**

Extend the resolver with a `max_distance_gap_m=1.0` ambiguity guard: if the two best candidates are within that distance and their confidence difference is below 0.05, raise `LookupError("ambiguous target")`. Add tests for both unique and ambiguous cases.

- [ ] **Step 5: Run and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/alignment -v`

Expected: all alignment tests pass.

```powershell
git add src/carla_driving/alignment tests/unit/alignment
git commit -m "feat: parse commands and resolve referenced scene targets"
```

### Task 10: Add the DriveMLM-style decision orchestrator and constrained VLM adapter

**Files:**
- Create: `src/carla_driving/alignment/vlm_adapter.py`
- Create: `src/carla_driving/decision/__init__.py`
- Create: `src/carla_driving/decision/orchestrator.py`
- Create: `tests/unit/decision/test_orchestrator.py`

- [ ] **Step 1: Test fast-path success, VLM fallback, and invalid model output**

```python
# tests/unit/decision/test_orchestrator.py
from carla_driving.decision.orchestrator import DecisionOrchestrator
from carla_driving.interfaces.commands import ASRCommand
from carla_driving.interfaces.intent import Action
from carla_driving.interfaces.scene import EgoState, SceneState, WeatherState


class FakeVlm:
    def decide(self, command, scene):
        return {"actions": [{"action": "STOP"}], "confidence": 0.88, "reason": "rain hazard"}


def scene():
    return SceneState(1, 1, EgoState(10.0, 1, 0.0), WeatherState(80.0, True), (), False, False, None)


def test_fast_path_does_not_call_vlm():
    result = DecisionOrchestrator(FakeVlm()).decide(ASRCommand("1", "停车", 0.99, 1), scene())
    assert result.actions[0].action is Action.STOP


def test_ambiguous_command_uses_vlm():
    result = DecisionOrchestrator(FakeVlm()).decide(ASRCommand("2", "前方情况不太对，小心一点", 0.96, 1), scene())
    assert result.reason == "rain hazard"
```

- [ ] **Step 2: Run and verify failures**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/decision/test_orchestrator.py -v`

Expected: FAIL because decision modules do not exist.

- [ ] **Step 3: Implement schema-only model output parsing**

```python
# src/carla_driving/alignment/vlm_adapter.py
import json
from typing import Protocol
from carla_driving.interfaces.commands import ASRCommand
from carla_driving.interfaces.intent import Action, ActionStep, DrivingIntent
from carla_driving.interfaces.scene import SceneState


class VlmDecisionModel(Protocol):
    def decide(self, command: ASRCommand, scene: SceneState) -> dict: ...


def intent_from_model(command_id: str, payload: dict) -> DrivingIntent:
    allowed = {item.value: item for item in Action}
    actions = []
    for raw in payload["actions"]:
        name = str(raw["action"])
        if name not in allowed:
            raise ValueError(f"unsupported model action: {name}")
        actions.append(ActionStep(
            allowed[name],
            target_id=raw.get("target_id"),
            target_speed_kmh=raw.get("target_speed_kmh"),
        ))
    return DrivingIntent(command_id, tuple(actions), float(payload["confidence"]), str(payload["reason"]))


def parse_json_object(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model did not return a JSON object")
    return json.loads(text[start:end + 1])
```

- [ ] **Step 4: Implement orchestration and enforce ASR confidence**

```python
# src/carla_driving/decision/orchestrator.py
from carla_driving.alignment.fast_parser import parse_fast
from carla_driving.alignment.vlm_adapter import intent_from_model
from carla_driving.interfaces.commands import ASRCommand
from carla_driving.interfaces.intent import Action, ActionStep, DrivingIntent
from carla_driving.interfaces.scene import SceneState


class DecisionOrchestrator:
    def __init__(self, vlm, minimum_asr_confidence: float = 0.75):
        self.vlm = vlm
        self.minimum_asr_confidence = minimum_asr_confidence

    def decide(self, command: ASRCommand, scene: SceneState) -> DrivingIntent:
        if command.confidence < self.minimum_asr_confidence:
            return DrivingIntent(command.command_id, (ActionStep(Action.STOP),), 1.0, "low ASR confidence")
        try:
            actions = parse_fast(command.text)
            return DrivingIntent(command.command_id, actions, command.confidence, "fast command path")
        except ValueError:
            return intent_from_model(command.command_id, self.vlm.decide(command, scene))
```

- [ ] **Step 5: Run and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/decision tests/unit/alignment -v`

Expected: all language/decision tests pass.

```powershell
git add src/carla_driving/alignment/vlm_adapter.py src/carla_driving/decision tests/unit/decision
git commit -m "feat: add constrained multimodal behavior decision path"
```

### Task 11: Benchmark the local 3B model before integrating it into the driving loop

**Files:**
- Create: `src/carla_driving/alignment/qwen_model.py`
- Create: `scripts/benchmark_vlm.ps1`
- Create: `tests/unit/alignment/test_vlm_output.py`

- [ ] **Step 1: Unit-test only the prompt and JSON contract with a fake generator**

```python
# tests/unit/alignment/test_vlm_output.py
from carla_driving.alignment.qwen_model import build_prompt


def test_prompt_limits_actions_and_requests_json():
    prompt = build_prompt("前方危险，小心一点", {"weather": {"rain": 80, "night": True}})
    assert "EMERGENCY_BRAKE" in prompt
    assert "JSON" in prompt
    assert "油门" not in prompt
```

- [ ] **Step 2: Run and verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/alignment/test_vlm_output.py -v`

Expected: FAIL because the Qwen adapter does not exist.

- [ ] **Step 3: Implement a bounded prompt and lazy-loading adapter**

```python
# src/carla_driving/alignment/qwen_model.py
import json
from carla_driving.alignment.vlm_adapter import parse_json_object


ACTIONS = "START, STOP, SET_SPEED, TURN_LEFT, TURN_RIGHT, CHANGE_LANE_LEFT, CHANGE_LANE_RIGHT, AVOID_OBJECT, EMERGENCY_BRAKE, RETURN_TO_LANE"


def build_prompt(command_text: str, scene: dict) -> str:
    return (
        "你是驾驶行为规划器。只从以下动作选择：" + ACTIONS + "。"
        "不要输出油门、刹车或方向盘连续值。安全优先。"
        "仅返回JSON：{\"actions\":[{\"action\":\"STOP\"}],\"confidence\":0.0,\"reason\":\"...\"}。"
        f"\n指令：{command_text}\n场景：{json.dumps(scene, ensure_ascii=False, separators=(',', ':'))}"
    )


class QwenDecisionModel:
    def __init__(self, model_path: str, image_provider, max_new_tokens: int = 64):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype="auto",
        )
        self.image_provider = image_provider
        self.max_new_tokens = max_new_tokens

    def decide(self, command, scene):
        image = self.image_provider(scene.frame_id)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": build_prompt(command.text, scene.to_dict())},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = output[:, inputs.input_ids.shape[1]:]
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return parse_json_object(decoded)
```

- [ ] **Step 4: Benchmark without CARLA, then with CARLA idle**

`scripts/benchmark_vlm.ps1` must invoke one warm-up and 30 measured inferences, use a fixed 640×384 image and fixed scene JSON, and write `artifacts/vlm-benchmark.json` containing model path, GPU name, peak allocated VRAM, mean latency, P95 latency, maximum latency, and valid-JSON rate.

Run first with CARLA stopped, then with `CarlaUE4.exe -quality-level=Low` running. The candidate is accepted only if it fits without out-of-memory errors, valid-JSON rate is 100%, and the measured P95 leaves enough of the 120ms vehicle-group budget for preprocessing and safety. If it fails, retain the adapter but keep complex commands in safe-stop mode until a smaller local model passes the same benchmark.

- [ ] **Step 5: Run unit tests and commit code, not generated benchmark data**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/alignment/test_vlm_output.py -v`

Expected: `1 passed`.

```powershell
git add src/carla_driving/alignment/qwen_model.py scripts/benchmark_vlm.ps1 tests/unit/alignment/test_vlm_output.py
git commit -m "feat: add benchmarked local multimodal model adapter"
```

**Phase 3 checkpoint:** clear commands complete through the fast path; ambiguous commands produce schema-valid high-level actions through a benchmarked local model or deterministically safe-stop; no free-form model text reaches vehicle control.

## Phase 4: Behavior execution, scenarios, and evaluation

### Task 12: Sequence high-level actions and map them to planner/controller goals

**Files:**
- Create: `src/carla_driving/decision/sequencer.py`
- Create: `src/carla_driving/planning/__init__.py`
- Create: `src/carla_driving/planning/route_planner.py`
- Create: `src/carla_driving/control/__init__.py`
- Create: `src/carla_driving/control/ego_agent.py`
- Create: `tests/unit/decision/test_sequencer.py`
- Create: `tests/unit/control/test_goal_mapping.py`

- [ ] **Step 1: Test that compound actions cannot skip their completion condition**

```python
# tests/unit/decision/test_sequencer.py
from carla_driving.decision.sequencer import ActionSequencer
from carla_driving.interfaces.intent import Action, ActionStep


def test_avoid_must_finish_before_speed_change():
    sequencer = ActionSequencer((
        ActionStep(Action.AVOID_OBJECT, target_id="ped-1"),
        ActionStep(Action.SET_SPEED, target_speed_kmh=20),
    ))
    assert sequencer.current.action is Action.AVOID_OBJECT
    sequencer.complete_current()
    assert sequencer.current.action is Action.SET_SPEED
```

- [ ] **Step 2: Run and verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/decision/test_sequencer.py -v`

Expected: FAIL because the sequencer is absent.

- [ ] **Step 3: Implement the minimal sequencer**

```python
# src/carla_driving/decision/sequencer.py
from collections import deque
from carla_driving.interfaces.intent import ActionStep


class ActionSequencer:
    def __init__(self, actions: tuple[ActionStep, ...]):
        if not actions:
            raise ValueError("actions must not be empty")
        self._remaining = deque(actions)
        self.current = self._remaining.popleft()
        self.finished = False

    def complete_current(self) -> None:
        if self._remaining:
            self.current = self._remaining.popleft()
        else:
            self.finished = True
```

- [ ] **Step 4: Wrap CARLA GlobalRoutePlanner and VehiclePIDController behind local interfaces**

`route_planner.py` must expose `trace_route(origin, destination)` and `lane_change(direction)` without leaking CARLA planner objects outside the module. `ego_agent.py` must map `STOP` and `EMERGENCY_BRAKE` to deterministic `carla.VehicleControl`, while speed/turn/lane/avoid actions produce planner goals consumed by CARLA's existing PID controller. Add fake-object tests asserting emergency brake produces `throttle=0.0`, `brake=1.0`, and that `SET_SPEED(20)` sets a 20km/h planner target rather than directly setting throttle.

- [ ] **Step 5: Run and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/decision/test_sequencer.py tests/unit/control/test_goal_mapping.py -v`

Expected: all sequencer/control mapping tests pass.

```powershell
git add src/carla_driving/decision/sequencer.py src/carla_driving/planning src/carla_driving/control tests/unit/decision/test_sequencer.py tests/unit/control/test_goal_mapping.py
git commit -m "feat: sequence behavior actions into CARLA planner goals"
```

### Task 13: Define six versioned scenario configurations and result criteria

**Files:**
- Create: `configs/scenarios/basic_control.json`
- Create: `configs/scenarios/turn_lane_change.json`
- Create: `configs/scenarios/pedestrian_avoid.json`
- Create: `configs/scenarios/construction_avoid.json`
- Create: `configs/scenarios/adverse_weather_stop.json`
- Create: `configs/scenarios/cut_in_emergency.json`
- Create: `tests/unit/evaluation/test_scenario_configs.py`

- [ ] **Step 1: Write config schema tests**

```python
# tests/unit/evaluation/test_scenario_configs.py
import json
from pathlib import Path


def test_six_scenario_configs_have_scoring_contracts():
    files = sorted(Path("configs/scenarios").glob("*.json"))
    assert len(files) == 6
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["id"] == path.stem
        assert data["max_duration_s"] > 0
        assert data["success_criteria"]
        assert "collision" in data["failure_criteria"]
        assert data["commands"]
```

- [ ] **Step 2: Run and verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/evaluation/test_scenario_configs.py -v`

Expected: FAIL because zero configs exist.

- [ ] **Step 3: Create all six configs from one strict schema**

Each JSON file must contain these exact keys:

```json
{
  "id": "pedestrian_avoid",
  "scenario_runner_type": "PedestrianCrossing",
  "town": "Town03",
  "weather": {"rain": 0.0, "night": false},
  "max_duration_s": 45,
  "commands": ["绕开前面的行人后加速到20公里每小时"],
  "success_criteria": {
    "route_completion_min": 0.95,
    "final_speed_kmh_min": 18.0,
    "target_alignment_required": true
  },
  "failure_criteria": {
    "collision": true,
    "wrong_lane_duration_s_max": 3.0,
    "timeout": true
  }
}
```

Use local custom orchestration for `basic_control`, `turn_lane_change`, and `adverse_weather_stop`; use ScenarioRunner `PedestrianCrossing`, `ConstructionObstacleTwoWays`, and `StaticCutIn` for the other three. Give each config at least five paraphrased command strings, including one low-confidence/ambiguous negative case in the test fixture rather than in the live command list.

- [ ] **Step 4: Run tests and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/evaluation/test_scenario_configs.py -v`

Expected: `1 passed`.

```powershell
git add configs/scenarios tests/unit/evaluation/test_scenario_configs.py
git commit -m "test: define six scored CARLA driving scenarios"
```

### Task 14: Add latency, completion, collision, and alignment metrics

**Files:**
- Create: `src/carla_driving/evaluation/__init__.py`
- Create: `src/carla_driving/evaluation/latency.py`
- Create: `src/carla_driving/evaluation/result.py`
- Create: `tests/unit/evaluation/test_metrics.py`

- [ ] **Step 1: Test percentile and aggregate metrics**

```python
# tests/unit/evaluation/test_metrics.py
from carla_driving.evaluation.latency import LatencyTracker
from carla_driving.evaluation.result import summarize_runs


def test_latency_reports_p95():
    tracker = LatencyTracker()
    for value in [10, 20, 30, 40, 50]:
        tracker.record_ms(value)
    assert tracker.summary()["p95_ms"] == 50.0


def test_completion_rate_and_alignment_accuracy():
    summary = summarize_runs([
        {"success": True, "aligned": True, "collision": False},
        {"success": False, "aligned": True, "collision": False},
    ])
    assert summary == {"completion_rate": 0.5, "alignment_accuracy": 1.0, "collision_rate": 0.0}
```

- [ ] **Step 2: Run and verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/evaluation/test_metrics.py -v`

Expected: FAIL because evaluation modules do not exist.

- [ ] **Step 3: Implement metrics with NumPy percentiles and zero-division guards**

```python
# src/carla_driving/evaluation/latency.py
import numpy as np


class LatencyTracker:
    def __init__(self):
        self.values = []

    def record_ms(self, value: float) -> None:
        if value < 0:
            raise ValueError("latency must be non-negative")
        self.values.append(float(value))

    def summary(self) -> dict:
        if not self.values:
            return {"mean_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        return {
            "mean_ms": float(np.mean(self.values)),
            "p95_ms": float(np.percentile(self.values, 95, method="higher")),
            "max_ms": max(self.values),
        }
```

```python
# src/carla_driving/evaluation/result.py
def summarize_runs(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("runs must not be empty")
    count = len(runs)
    return {
        "completion_rate": sum(bool(item["success"]) for item in runs) / count,
        "alignment_accuracy": sum(bool(item["aligned"]) for item in runs) / count,
        "collision_rate": sum(bool(item["collision"]) for item in runs) / count,
    }
```

- [ ] **Step 4: Run and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/evaluation -v`

Expected: all evaluation tests pass.

```powershell
git add src/carla_driving/evaluation tests/unit/evaluation
git commit -m "feat: measure driving completion alignment and latency"
```

### Task 15: Build the end-to-end agent runner with failure-safe lifecycle

**Files:**
- Create: `src/carla_driving/cli/__init__.py`
- Create: `src/carla_driving/cli/run_agent.py`
- Create: `configs/runtime.json`
- Create: `tests/integration/test_decision_pipeline.py`
- Create: `tests/smoke/test_end_to_end_scenario.py`

- [ ] **Step 1: Write a CARLA-free integration test with fake sensors and model**

```python
# tests/integration/test_decision_pipeline.py
from carla_driving.decision.orchestrator import DecisionOrchestrator
from carla_driving.interfaces.commands import ASRCommand
from carla_driving.interfaces.intent import Action
from carla_driving.interfaces.scene import EgoState, SceneState, WeatherState
from carla_driving.safety.arbiter import SafetyArbiter


class NeverCalledVlm:
    def decide(self, command, scene):
        raise AssertionError("fast command must not call VLM")


def test_stop_command_survives_decision_and_safety():
    scene = SceneState(1, 1, EgoState(10, 1, 0), WeatherState(0, False), (), True, True, None)
    command = ASRCommand("stop-1", "停车", 0.99, 1)
    intent = DecisionOrchestrator(NeverCalledVlm()).decide(command, scene)
    safe = SafetyArbiter().arbitrate(intent, scene)
    assert safe.actions[0].action is Action.STOP
```

- [ ] **Step 2: Run and verify the integration baseline**

Run: `.\.venv\Scripts\python.exe -m pytest tests/integration/test_decision_pipeline.py -v`

Expected: `1 passed`; if it fails, repair earlier contracts before creating the live runner.

- [ ] **Step 3: Create runtime configuration and lifecycle order**

```json
{
  "carla": {"host": "127.0.0.1", "port": 2000, "timeout_s": 5.0, "fixed_delta_s": 0.05},
  "sensors": {"rgb_width": 640, "rgb_height": 384, "rgb_hz": 10, "lidar_hz": 10},
  "asr": {"minimum_confidence": 0.75, "maximum_age_ms": 1000},
  "safety": {"emergency_ttc_s": 1.5, "model_timeout_ms": 100},
  "logging": {"directory": "artifacts/runs", "jsonl": true}
}
```

`run_agent.py` must perform this exact lifecycle:

1. Load and validate runtime/scenario JSON.
2. Connect to CARLA and capture original settings.
3. Enter synchronous mode and spawn ego/sensors.
4. Start ScenarioRunner or local scenario orchestration.
5. Build frame-aligned `SceneState` values.
6. Receive one `ASRCommand` from JSON stdin or a named pipe adapter.
7. Run fast/VLM decision with a hard timeout.
8. Apply safety arbitration.
9. Sequence planner goals and apply PID control.
10. Write JSONL stage timestamps and result criteria.
11. On any exception, apply full brake before destroying actors.
12. Restore original CARLA settings in `finally`.

- [ ] **Step 4: Add one live basic-stop smoke test**

`tests/smoke/test_end_to_end_scenario.py` must start the runner against an already-running CARLA server, send a `STOP` command, and assert that the logged final control contains `brake >= 0.9`, no collision event, and a restored synchronous setting after shutdown. Gate it behind `CARLA_SMOKE=1`.

- [ ] **Step 5: Run unit/integration/smoke suites and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit tests/integration -v`

Expected: all tests pass.

With CARLA running:

```powershell
$env:CARLA_SMOKE="1"
.\.venv\Scripts\python.exe -m pytest tests/smoke/test_end_to_end_scenario.py -v
```

Expected: `1 passed` and a JSONL run artifact.

```powershell
git add src/carla_driving/cli configs/runtime.json tests/integration tests/smoke/test_end_to_end_scenario.py
git commit -m "feat: run failure-safe multimodal CARLA agent loop"
```

### Task 16: Normalize CARLA, NuScenes, and Waymo representative records

**Files:**
- Create: `src/carla_driving/data/__init__.py`
- Create: `src/carla_driving/data/record.py`
- Create: `src/carla_driving/data/adapters.py`
- Create: `src/carla_driving/data/split.py`
- Create: `tests/fixtures/data/carla_sample.json`
- Create: `tests/fixtures/data/nuscenes_sample.json`
- Create: `tests/fixtures/data/waymo_sample.json`
- Create: `tests/unit/data/test_adapters.py`
- Create: `tests/unit/data/test_split.py`

- [ ] **Step 1: Write adapter and leakage tests**

```python
# tests/unit/data/test_adapters.py
import json
from pathlib import Path
import pytest
from carla_driving.data.adapters import normalize_record


@pytest.mark.parametrize("source", ["carla", "nuscenes", "waymo"])
def test_representative_sample_normalizes(source):
    raw = json.loads(Path(f"tests/fixtures/data/{source}_sample.json").read_text(encoding="utf-8"))
    record = normalize_record(source, raw)
    assert record.source == source
    assert record.scene_id
    assert record.timestamp_us >= 0
    assert record.rgb_ref
    assert record.lidar_ref
    assert record.ego_speed_mps >= 0
```

```python
# tests/unit/data/test_split.py
from carla_driving.data.record import MultimodalRecord
from carla_driving.data.split import split_by_scene


def record(scene):
    return MultimodalRecord("carla", scene, 1, "rgb.png", "lidar.npy", 1.0, "停车", "STOP")


def test_scene_never_leaks_between_train_and_test():
    train, test = split_by_scene([record("a"), record("a"), record("b"), record("c")], test_ratio=0.34)
    assert {item.scene_id for item in train}.isdisjoint({item.scene_id for item in test})
```

- [ ] **Step 2: Run and verify missing-module failures**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/data -v`

Expected: FAIL because the data package and fixtures do not exist.

- [ ] **Step 3: Implement the normalized record and source field maps**

```python
# src/carla_driving/data/record.py
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MultimodalRecord:
    source: str
    scene_id: str
    timestamp_us: int
    rgb_ref: str
    lidar_ref: str
    ego_speed_mps: float
    command_text: str
    expected_action: str
```

```python
# src/carla_driving/data/adapters.py
from .record import MultimodalRecord


FIELD_MAPS = {
    "carla": {"scene": "scene_id", "time": "timestamp_us", "rgb": "rgb_path", "lidar": "lidar_path", "speed": "speed_mps"},
    "nuscenes": {"scene": "scene_token", "time": "timestamp", "rgb": "cam_front", "lidar": "lidar_top", "speed": "ego_speed_mps"},
    "waymo": {"scene": "context_name", "time": "frame_timestamp_micros", "rgb": "front_image", "lidar": "top_lidar", "speed": "vehicle_speed_mps"},
}


def normalize_record(source: str, raw: dict) -> MultimodalRecord:
    if source not in FIELD_MAPS:
        raise ValueError(f"unsupported source: {source}")
    fields = FIELD_MAPS[source]
    return MultimodalRecord(
        source=source,
        scene_id=str(raw[fields["scene"]]),
        timestamp_us=int(raw[fields["time"]]),
        rgb_ref=str(raw[fields["rgb"]]),
        lidar_ref=str(raw[fields["lidar"]]),
        ego_speed_mps=float(raw[fields["speed"]]),
        command_text=str(raw["command_text"]),
        expected_action=str(raw["expected_action"]),
    )
```

```python
# src/carla_driving/data/split.py
from .record import MultimodalRecord


def split_by_scene(records: list[MultimodalRecord], test_ratio: float = 0.2):
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1")
    scenes = sorted({item.scene_id for item in records})
    test_count = max(1, round(len(scenes) * test_ratio))
    test_scenes = set(scenes[-test_count:])
    train = [item for item in records if item.scene_id not in test_scenes]
    test = [item for item in records if item.scene_id in test_scenes]
    return train, test
```

- [ ] **Step 4: Create three deterministic representative fixtures**

Each fixture must contain its source-specific scene, timestamp, front RGB, top LiDAR, and speed keys plus `command_text` and `expected_action`. Use repository-local fictitious paths such as `fixtures/rgb/0001.png`; do not copy media or proprietary annotations into Git.

- [ ] **Step 5: Run, document source licenses, and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/data -v`

Expected: all data adapter and split tests pass.

Add dataset source/version/license notes to `docs/references/lmdrive-mapping.md`, then commit:

```powershell
git add src/carla_driving/data tests/fixtures/data tests/unit/data docs/references/lmdrive-mapping.md
git commit -m "feat: normalize multimodal driving dataset records"
```

### Task 17: Execute the six-scenario, 20-run acceptance matrix and freeze the demo

**Files:**
- Create: `scripts/run_acceptance.ps1`
- Create: `docs/acceptance/basic-track-report.md`
- Create: `tests/acceptance/test_report_thresholds.py`

- [ ] **Step 1: Write report threshold validation before running acceptance**

```python
# tests/acceptance/test_report_thresholds.py
import json
from pathlib import Path


def test_acceptance_report_meets_internal_targets():
    report = json.loads(Path("artifacts/acceptance/summary.json").read_text(encoding="utf-8"))
    assert report["runs_per_scenario"] >= 20
    assert report["completion_rate"] >= 0.95
    assert report["alignment_accuracy"] >= 0.99
    assert report["collision_rate"] == 0.0
    assert report["fast_parse_p95_ms"] <= 40.0
    assert report["vehicle_decision_p95_ms"] <= 120.0
```

- [ ] **Step 2: Verify the test fails before acceptance artifacts exist**

Run: `.\.venv\Scripts\python.exe -m pytest tests/acceptance/test_report_thresholds.py -v`

Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 3: Implement the acceptance runner**

`scripts/run_acceptance.ps1` must iterate the six config files in lexical order, run each scenario 20 times with fixed seeds `1000..1019`, stop on infrastructure failure after recording it separately, merge JSONL outputs into `artifacts/acceptance/summary.json`, and preserve per-run logs. It must not count CARLA startup failure as a driving collision, but the report must still show the incomplete run.

- [ ] **Step 4: Run acceptance, analyze failures, and rerun the full matrix after every fix**

Run: `powershell -ExecutionPolicy Bypass -File .\scripts\run_acceptance.ps1`

Then run: `.\.venv\Scripts\python.exe -m pytest tests/acceptance/test_report_thresholds.py -v`

Expected: `1 passed`. Partial reruns are diagnostic only and cannot establish release readiness.

- [ ] **Step 5: Write the human-readable report and freeze commit**

`docs/acceptance/basic-track-report.md` must include hardware/software versions, ScenarioRunner commit, model/checkpoint identifier, six per-scenario tables, mean/P95/max latency, perception/target-alignment errors, collisions, infrastructure failures, and exact reproduction commands.

```powershell
git add scripts/run_acceptance.ps1 tests/acceptance/test_report_thresholds.py docs/acceptance/basic-track-report.md
git commit -m "test: verify basic-track multimodal driving acceptance"
git tag basic-track-demo-v1
```

## Final verification commands

Run all commands from the repository root with CARLA stopped unless noted:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit tests/integration -v
git diff --check
git status --short
```

Then start CARLA in low-quality mode and run:

```powershell
$env:CARLA_SMOKE="1"
.\.venv\Scripts\python.exe -m pytest tests/smoke -v
powershell -ExecutionPolicy Bypass -File .\scripts\run_acceptance.ps1
.\.venv\Scripts\python.exe -m pytest tests/acceptance/test_report_thresholds.py -v
```

Release evidence requires fresh output from all four suites. Do not infer acceptance from individual scenario videos or previously generated summaries.

## Spec coverage self-review

| Specification area | Implementation tasks |
|---|---|
| Single-machine offline Python/CARLA baseline | Tasks 1, 4, 15 |
| ScenarioRunner v0.9.16 and open-source study limits | Tasks 3, 5 |
| RGB/LiDAR perception and scene state | Tasks 7, 8 |
| ASR contract and multimodal target alignment | Tasks 2, 9, 10 |
| DriveMLM-style behavior state boundary | Tasks 3, 10, 12 |
| Deterministic safety and minimum-risk fallback | Tasks 6, 10, 15 |
| Route planning, PID, compound actions | Task 12 |
| Six standard scenarios | Task 13 |
| Latency, completion, collision, alignment metrics | Tasks 14, 16 |
| Model VRAM/latency benchmark | Task 11 |
| CARLA/NuScenes/Waymo unified record format | Task 16 |
| Repeatable 20-run regression and demo freeze | Tasks 15, 17 |

All implementation work must preserve the user's existing uncommitted `requirements.txt`, `startCarla.txt`, CARLA binaries, and AdditionalMaps files unless a later task explicitly requires and scopes a change.
