
import inspect

from navigation.live_expert_episode import LiveExpertEpisode


def test_collision_checked_before_command_interval():
    source = inspect.getsource(LiveExpertEpisode.run)

    collision_index = source.index('if collision:')
    interval_index = source.index('now - self.last_command_time')

    assert collision_index < interval_index


def test_sensor_timeout_checked_before_command_interval():
    source = inspect.getsource(LiveExpertEpisode.run)

    sensor_index = source.index('sensor_timeout =')
    interval_index = source.index('now - self.last_command_time')

    assert sensor_index < interval_index


def test_collision_and_sensor_loss_stop_robot():
    source = inspect.getsource(LiveExpertEpisode.run)

    assert 'self.navigator.terminate("COLLISION")' in source
    assert 'self.navigator.terminate("STALE_SENSORS")' in source
    assert 'self.robot.stop()' in source
