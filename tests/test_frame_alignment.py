from navigation.frame_alignment import WorldAlignedPlanner


class RecordingPlanner:
    def __init__(self, path):
        self.path = path
        self.planned = None
        self.validated = None

    def plan(self, start, goal):
        self.planned = (start, goal)
        return self.path

    def validate_path(self, waypoints):
        self.validated = list(waypoints)
        return True


def test_identical_frames_keep_coordinates():
    inner = RecordingPlanner([(0.0, -1.5), (2.5, -1.5)])
    aligned = WorldAlignedPlanner(
        inner,
        world_pose=(0.0, 0.0, 0.0),
        odom_pose=(0.0, 0.0, 0.0),
    )

    path = aligned.plan((0.0, -1.5), (2.5, -1.5))

    assert inner.planned == ((0.0, -1.5), (2.5, -1.5))
    assert path == [(0.0, -1.5), (2.5, -1.5)]


def test_teleport_plans_on_the_world_obstacle_map():
    """Odometry stays at the origin after a world teleport to y=-1.5."""

    inner = RecordingPlanner([(0.0, -1.5), (2.5, -1.5)])
    aligned = WorldAlignedPlanner(
        inner,
        world_pose=(0.0, -1.5, 0.0),
        odom_pose=(0.0, 0.0, 0.0),
    )

    path = aligned.plan((0.0, 0.0), (2.5, 0.0))

    assert inner.planned[0] == (0.0, -1.5)
    assert inner.planned[1] == (2.5, -1.5)
    assert path[0] == (0.0, 0.0)
    assert path[-1] == (2.5, 0.0)
    assert aligned.validate_path(path) is True
    assert inner.validated[0] == (0.0, -1.5)
    assert inner.validated[-1] == (2.5, -1.5)
