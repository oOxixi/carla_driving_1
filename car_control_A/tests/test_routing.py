from car_control_A.routing import RouteReference, LateralController


def test_route_reference_and_lateral_protocol_are_minimal() -> None:
    route = RouteReference(points_xy_m=((0.0, 0.0), (1.0, 0.0)), curvature_per_m=0.1, target_speed_mps=5.0)
    assert route.target_speed_mps == 5.0

    class FixedSteer:
        def steer(self, reference: RouteReference) -> float:
            return 0.0

    assert isinstance(FixedSteer(), LateralController)
