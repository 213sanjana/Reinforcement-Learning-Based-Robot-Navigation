"""Pure readiness state for the stationary diagnostic; no ROS/process imports."""
import math

CONTROLLERS = frozenset({'joint_state_broadcaster', 'platform_velocity_controller'})
SENSORS = ('clock', 'odometry', 'lidar')


class ReadinessGate:
    """Use monotonic receive/advance ages and structured controller states.

    The caller alone owns service timeouts and shutdown. This object never
    sends commands, requests, signals, or retries. Terminal failures latch.
    """
    def __init__(self, started, startup_timeout=150.0, live_timeout=180.0,
                 observation_duration=30.0, freshness=0.5):
        params = (started, startup_timeout, live_timeout, observation_duration, freshness)
        if not all(math.isfinite(v) for v in params) or min(params[1:]) <= 0:
            raise ValueError('Finite times and positive limits required')
        self.started = started
        self.startup_timeout = startup_timeout
        self.live_timeout = live_timeout
        self.observation_duration = observation_duration
        self.freshness = freshness
        self.manager_responded = False
        self.active = set()
        self.sensors = {}
        self.ready_at = None
        self.failure = None
        self.stopping = False
        self.last_now = started

    def controller_response(self, response):
        if self.stopping or self.failure:
            return
        controllers = response.get('controller') if isinstance(response, dict) else None
        if not isinstance(controllers, list):
            self.fail('controller_manager: malformed list_controllers response')
            return
        self.manager_responded = True
        self.active = {c.get('name') for c in controllers
                       if isinstance(c, dict) and c.get('state') == 'active'}

    def ingest(self, name, stamp_ns, received, valid=True):
        if self.stopping or self.failure:
            return
        if name not in SENSORS:
            return
        if (not isinstance(stamp_ns, int) or isinstance(stamp_ns, bool)
                or stamp_ns < 0 or not math.isfinite(received)):
            self.fail(name + ': invalid timestamp')
            return
        old = self.sensors.get(name)
        if old and (stamp_ns < old['stamp'] or received < old['received']):
            self.fail(name + ': timestamp moved backwards')
            return
        advanced = old is not None and stamp_ns > old['stamp']
        self.sensors[name] = {
            'stamp': stamp_ns, 'received': received, 'valid': bool(valid),
            'advances': (old['advances'] if old else 0) + int(advanced),
            'last_advance': received if advanced else (old['last_advance'] if old else received),
        }

    def fail(self, reason):
        if not self.stopping and self.failure is None:
            self.failure = reason

    def begin_shutdown(self):
        self.stopping = True

    def evaluate(self, now, launch_running):
        if self.stopping:
            return {'status': 'stopping', 'reason': 'shutdown in progress', 'ready_at': self.ready_at}
        if not math.isfinite(now) or now < self.last_now:
            self.fail('diagnostic monotonic clock invalid')
        self.last_now = now
        missing = []
        if not launch_running:
            missing.append('launch: not running')
        if not self.manager_responded:
            missing.append('controller_manager: no successful service response')
        inactive = CONTROLLERS - self.active
        if inactive:
            missing.append('controllers: not active: ' + ', '.join(sorted(inactive)))
        for name in SENSORS:
            data = self.sensors.get(name)
            if data is None:
                missing.append(name + ': missing')
            elif not data['valid']:
                missing.append(name + ': invalid data/frame')
            elif data['advances'] < 1:
                missing.append(name + ': acquisition timestamp has not advanced')
            elif not (0 <= now-data['received'] <= self.freshness
                      and 0 <= now-data['last_advance'] <= self.freshness):
                missing.append(name + ': stale receive/advance age')
            elif name != 'clock' and 'clock' in self.sensors:
                lag = (self.sensors['clock']['stamp'] - data['stamp']) / 1e9
                if abs(lag) > self.freshness:
                    missing.append(name + ': acquisition stamp disagrees with /clock')
        conditions = {
            'launch_running': bool(launch_running),
            'controller_manager_responded': self.manager_responded,
            'required_controllers_active': not inactive,
            'sensors_fresh_and_advancing': not any(
                m.startswith(tuple(name + ':' for name in SENSORS)) for m in missing),
        }
        if not self.failure:
            if now-self.started >= self.live_timeout:
                self.fail('overall live deadline')
            elif self.ready_at is not None and missing:
                self.fail('readiness lost: ' + '; '.join(missing))
            elif self.ready_at is None and now-self.started >= self.startup_timeout:
                self.fail('startup timeout: ' + '; '.join(missing))
        if self.failure:
            return {'status': 'failed', 'reason': self.failure, 'missing': missing, 'conditions': conditions, 'ready_at': self.ready_at}
        if missing:
            return {'status': 'waiting', 'reason': '; '.join(missing), 'missing': missing, 'conditions': conditions, 'ready_at': None}
        if self.ready_at is None:
            self.ready_at = now
        complete = now-self.ready_at >= self.observation_duration
        return {'status': 'complete' if complete else 'ready',
                'reason': 'observation window complete' if complete else 'all structured conditions satisfied',
                'conditions': conditions, 'ready_at': self.ready_at, 'observation_elapsed': now-self.ready_at}
