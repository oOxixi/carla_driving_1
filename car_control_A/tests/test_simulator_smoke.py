from __future__ import annotations

import os

import pytest

from car_control_A.simulator import CarlaSession


@pytest.mark.skipif(os.environ.get("CARLA_SMOKE") != "1", reason="set CARLA_SMOKE=1 with CARLA server running")
def test_carla_session_spawns_ego_and_restores_world_tm_and_actor() -> None:
    """Optional live smoke test; it never launches or loads a CARLA server."""
    import carla

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    original = world.get_settings()
    traffic_manager = client.get_trafficmanager(8000)
    # CARLA exposes no TM getter.  The project server-launch contract starts
    # it asynchronous, so this test explicitly restores that documented mode.
    ego = None
    with CarlaSession(world, traffic_manager=traffic_manager,
                      tm_previous_synchronous_mode=False,
                      fixed_delta_seconds=0.05) as session:
        blueprint = world.get_blueprint_library().filter("vehicle.*")[0]
        for transform in world.get_map().get_spawn_points():
            candidate = world.try_spawn_actor(blueprint, transform)
            if candidate is not None:
                ego = session.track_actor(candidate)
                break
        if ego is None:
            pytest.skip("no free CARLA vehicle spawn point")
        assert session.tick(timeout_s=10.0) > 0
        assert ego.is_alive
    restored = world.get_settings()
    assert restored.synchronous_mode == original.synchronous_mode
    assert restored.fixed_delta_seconds == original.fixed_delta_seconds
    assert ego is not None and not ego.is_alive
