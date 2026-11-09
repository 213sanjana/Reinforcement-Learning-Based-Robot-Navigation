"""Guarded factory for expert navigation components.

Importing this module does not initialize ROS or Gazebo.
"""

from jackal_rl_navigation.collision_config import CollisionConfig


def create_live_runner(
    episode,
    *,
    world_path,
    inflation_radius,
    allow_live=False,
    environment_factory=None,
    planner_factory=None,
    runner_factory=None,
):
    """Create a runner only after explicit authorization."""

    if not allow_live:
        raise RuntimeError(
            "Live runner creation is disabled. "
            "No robot movement or simulation reset was requested."
        )

    if environment_factory is None:
        from jackal_rl_navigation.gazebo_env import JackalGazeboEnv
        environment_factory = JackalGazeboEnv

    if planner_factory is None:
        from planners.gazebo_astar import GazeboAStarPlanner
        planner_factory = GazeboAStarPlanner

    if runner_factory is None:
        from navigation.live_expert_episode import LiveExpertEpisode
        runner_factory = LiveExpertEpisode

    collision_config = CollisionConfig(enabled=True)

    env = None

    try:
        env = environment_factory(
            collision_config=collision_config,
        )

        planner = planner_factory(
            world_path=world_path,
            inflation_radius=inflation_radius,
        )

        runner = runner_factory(
            env=env,
            planner=planner,
            controller_type="pure-pursuit",
            enable_motion=True,
            command_interval=0.1,
        )

        return runner

    except BaseException:
        if env is not None:
            env.close()
        raise
