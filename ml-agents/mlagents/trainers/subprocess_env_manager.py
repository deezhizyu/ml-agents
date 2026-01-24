"""
Subprocess environment manager for parallel Unity environment execution.

Security Note - Cloudpickle Usage:
    This module uses cloudpickle to serialize environment factory functions for
    multiprocessing. This is safe in this context because:

    1. Factory functions are defined internally in trusted code, never from user input
    2. Subprocess workers operate in a controlled training environment
    3. No network communication of pickled data occurs
    4. Used exclusively for trusted code execution in training contexts
    5. The pickled data never crosses security boundaries

    Alternative approaches were considered (multiprocessing.spawn with importable
    factories, shared memory via env_manager_shared_memory.py), but cloudpickle
    provides the best balance of flexibility and performance for this use case.
"""
import datetime
from typing import Dict, NamedTuple, List, Any, Optional, Callable, Set
import cloudpickle
import enum
import time

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import (
    UnityCommunicationException,
    UnityTimeOutException,
    UnityEnvironmentException,
    UnityCommunicatorStoppedException,
)
from multiprocessing import Process, Pipe, Queue
from multiprocessing.connection import Connection
from queue import Empty as EmptyQueueException
from mlagents_envs.base_env import BaseEnv, BehaviorName, BehaviorSpec
from mlagents_envs import logging_util
from mlagents.trainers.env_manager import EnvManager, EnvironmentStep, AllStepResult
from mlagents.trainers.settings import TrainerSettings
from mlagents_envs.timers import (
    TimerNode,
    timed,
    hierarchical_timer,
    reset_timers,
    get_timer_root,
)
from mlagents.trainers.settings import ParameterRandomizationSettings, RunOptions
from mlagents.trainers.action_info import ActionInfo
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
    EngineConfig,
)
from mlagents_envs.side_channel.stats_side_channel import (
    EnvironmentStats,
    StatsSideChannel,
)
from mlagents.trainers.training_analytics_side_channel import (
    TrainingAnalyticsSideChannel,
)
from mlagents_envs.side_channel.side_channel import SideChannel

logger = logging_util.get_logger(__name__)
WORKER_SHUTDOWN_TIMEOUT_S = 10


class EnvironmentCommand(enum.Enum):
    STEP = 1
    BEHAVIOR_SPECS = 2
    ENVIRONMENT_PARAMETERS = 3
    RESET = 4
    CLOSE = 5
    ENV_EXITED = 6
    CLOSED = 7
    TRAINING_STARTED = 8


class EnvironmentRequest(NamedTuple):
    cmd: EnvironmentCommand
    payload: Any = None


class EnvironmentResponse(NamedTuple):
    cmd: EnvironmentCommand
    worker_id: int
    payload: Any


class StepResponse(NamedTuple):
    all_step_result: AllStepResult
    timer_root: Optional[TimerNode]
    environment_stats: EnvironmentStats


class UnityEnvWorker:
    def __init__(self, process: Process, worker_id: int, conn: Connection):
        self.process = process
        self.worker_id = worker_id
        self.conn = conn
        self.previous_step: EnvironmentStep = EnvironmentStep.empty(worker_id)
        self.previous_all_action_info: Dict[str, ActionInfo] = {}
        self.waiting = False
        self.closed = False

    def send(self, cmd: EnvironmentCommand, payload: Any = None) -> None:
        try:
            req = EnvironmentRequest(cmd, payload)
            self.conn.send(req)
        except (BrokenPipeError, EOFError):
            raise UnityCommunicationException("UnityEnvironment worker: send failed.")

    def recv(self) -> EnvironmentResponse:
        try:
            response: EnvironmentResponse = self.conn.recv()
            if response.cmd == EnvironmentCommand.ENV_EXITED:
                env_exception: Exception = response.payload
                raise env_exception
            return response
        except (BrokenPipeError, EOFError):
            raise UnityCommunicationException("UnityEnvironment worker: recv failed.")

    def request_close(self):
        try:
            self.conn.send(EnvironmentRequest(EnvironmentCommand.CLOSE))
        except (BrokenPipeError, EOFError):
            logger.debug(
                f"UnityEnvWorker {self.worker_id} got exception trying to close."
            )
            pass


def _handle_step_command(
    env: UnityEnvironment,
    req: EnvironmentRequest,
    worker_id: int,
    step_queue: Queue,
    stats_channel: StatsSideChannel,
) -> None:
    """Handle STEP command by executing environment step and returning results."""
    all_action_info = req.payload
    for brain_name, action_info in all_action_info.items():
        if len(action_info.agent_ids) > 0:
            env.set_actions(brain_name, action_info.env_action)

    env.step()

    # Generate results from all behaviors
    all_step_result: AllStepResult = {}
    for brain_name in env.behavior_specs:
        all_step_result[brain_name] = env.get_steps(brain_name)

    # Collect statistics and timing information
    env_stats = stats_channel.get_and_reset_stats()
    step_response = StepResponse(all_step_result, get_timer_root(), env_stats)

    step_queue.put(EnvironmentResponse(EnvironmentCommand.STEP, worker_id, step_response))
    reset_timers()


def _handle_reset_command(
    env: UnityEnvironment,
    parent_conn: Connection,
    worker_id: int,
) -> None:
    """Handle RESET command by resetting environment and returning initial state."""
    env.reset()
    all_step_result: AllStepResult = {}
    for brain_name in env.behavior_specs:
        all_step_result[brain_name] = env.get_steps(brain_name)
    parent_conn.send(EnvironmentResponse(EnvironmentCommand.RESET, worker_id, all_step_result))


def _handle_environment_parameters(
    req: EnvironmentRequest,
    env_parameters: EnvironmentParametersChannel,
) -> None:
    """Handle ENVIRONMENT_PARAMETERS command by applying parameter randomization."""
    for k, v in req.payload.items():
        if isinstance(v, ParameterRandomizationSettings):
            v.apply(k, env_parameters)


def _handle_training_started(
    req: EnvironmentRequest,
    training_analytics_channel: Optional[TrainingAnalyticsSideChannel],
) -> None:
    """Handle TRAINING_STARTED command by notifying analytics channel."""
    if training_analytics_channel:
        behavior_name, trainer_config = req.payload
        training_analytics_channel.training_started(behavior_name, trainer_config)


def _initialize_worker_environment(
    pickled_env_factory: str,
    worker_id: int,
    run_options: RunOptions,
    log_level: int,
) -> tuple[
    UnityEnvironment,
    EnvironmentParametersChannel,
    StatsSideChannel,
    Optional[TrainingAnalyticsSideChannel],
]:
    """
    Initialize the worker subprocess environment with necessary channels.

    Returns:
        Tuple of (env, env_parameters, stats_channel, training_analytics_channel)
    """
    # Set log level for this subprocess
    logging_util.set_log_level(log_level)

    # Deserialize environment factory (see module docstring for security considerations)
    env_factory: Callable[[int, List[SideChannel]], UnityEnvironment] = (
        cloudpickle.loads(pickled_env_factory)
    )

    # Initialize side channels
    env_parameters = EnvironmentParametersChannel()
    stats_channel = StatsSideChannel()

    # Training analytics only enabled for worker 0
    training_analytics_channel: Optional[TrainingAnalyticsSideChannel] = None
    if worker_id == 0:
        training_analytics_channel = TrainingAnalyticsSideChannel()

    # Configure engine settings
    engine_config = EngineConfig(
        width=run_options.engine_settings.width,
        height=run_options.engine_settings.height,
        quality_level=run_options.engine_settings.quality_level,
        time_scale=run_options.engine_settings.time_scale,
        target_frame_rate=run_options.engine_settings.target_frame_rate,
        capture_frame_rate=run_options.engine_settings.capture_frame_rate,
    )
    engine_configuration_channel = EngineConfigurationChannel()
    engine_configuration_channel.set_configuration(engine_config)

    # Create environment with all channels
    side_channels = [env_parameters, engine_configuration_channel, stats_channel]
    if training_analytics_channel is not None:
        side_channels.append(training_analytics_channel)

    env = env_factory(worker_id, side_channels)

    # Disable training analytics if environment doesn't support it
    if (
        not env.academy_capabilities
        or not env.academy_capabilities.trainingAnalytics
    ):
        training_analytics_channel = None

    if training_analytics_channel:
        training_analytics_channel.environment_initialized(run_options)

    return env, env_parameters, stats_channel, training_analytics_channel


def worker(
    parent_conn: Connection,
    step_queue: Queue,
    pickled_env_factory: str,
    worker_id: int,
    run_options: RunOptions,
    log_level: int = logging_util.INFO,
) -> None:
    """
    Worker subprocess for parallel Unity environment execution.

    This function runs in a separate process and handles environment commands
    via inter-process communication. It manages a Unity environment instance
    and executes step/reset/close operations as requested.

    Args:
        parent_conn: Connection for receiving commands and sending responses
        step_queue: Queue for sending step results back to main process
        pickled_env_factory: Serialized environment factory function
        worker_id: Unique identifier for this worker
        run_options: Training run configuration
        log_level: Logging level for this subprocess
    """
    env: Optional[UnityEnvironment] = None

    try:
        # Initialize environment and channels
        env, env_parameters, stats_channel, training_analytics_channel = (
            _initialize_worker_environment(
                pickled_env_factory, worker_id, run_options, log_level
            )
        )

        # Main command processing loop
        while True:
            req: EnvironmentRequest = parent_conn.recv()

            if req.cmd == EnvironmentCommand.STEP:
                _handle_step_command(env, req, worker_id, step_queue, stats_channel)

            elif req.cmd == EnvironmentCommand.BEHAVIOR_SPECS:
                parent_conn.send(
                    EnvironmentResponse(EnvironmentCommand.BEHAVIOR_SPECS, worker_id, env.behavior_specs)
                )

            elif req.cmd == EnvironmentCommand.ENVIRONMENT_PARAMETERS:
                _handle_environment_parameters(req, env_parameters)

            elif req.cmd == EnvironmentCommand.TRAINING_STARTED:
                _handle_training_started(req, training_analytics_channel)

            elif req.cmd == EnvironmentCommand.RESET:
                _handle_reset_command(env, parent_conn, worker_id)

            elif req.cmd == EnvironmentCommand.CLOSE:
                break

    except (
        KeyboardInterrupt,
        UnityCommunicationException,
        UnityTimeOutException,
        UnityEnvironmentException,
        UnityCommunicatorStoppedException,
    ) as ex:
        logger.debug(f"UnityEnvironment worker {worker_id}: environment stopping.")
        step_queue.put(EnvironmentResponse(EnvironmentCommand.ENV_EXITED, worker_id, ex))
        parent_conn.send(EnvironmentResponse(EnvironmentCommand.ENV_EXITED, worker_id, ex))

    except Exception as ex:
        logger.exception(
            f"UnityEnvironment worker {worker_id}: environment raised an unexpected exception."
        )
        step_queue.put(EnvironmentResponse(EnvironmentCommand.ENV_EXITED, worker_id, ex))
        parent_conn.send(EnvironmentResponse(EnvironmentCommand.ENV_EXITED, worker_id, ex))

    finally:
        logger.debug(f"UnityEnvironment worker {worker_id} closing.")
        if env is not None:
            env.close()
        logger.debug(f"UnityEnvironment worker {worker_id} done.")
        parent_conn.close()
        step_queue.put(EnvironmentResponse(EnvironmentCommand.CLOSED, worker_id, None))
        step_queue.close()


class SubprocessEnvManager(EnvManager):
    def __init__(
        self,
        env_factory: Callable[[int, List[SideChannel]], BaseEnv],
        run_options: RunOptions,
        n_env: int = 1,
    ):
        super().__init__()
        self.env_workers: List[UnityEnvWorker] = []
        self.step_queue: Queue = Queue()
        self.workers_alive = 0
        self.env_factory = env_factory
        self.run_options = run_options
        self.env_parameters: Optional[Dict] = None
        # Each worker is correlated with a list of times they restarted within the last time period.
        self.recent_restart_timestamps: List[List[datetime.datetime]] = [
            [] for _ in range(n_env)
        ]
        self.restart_counts: List[int] = [0] * n_env
        for worker_idx in range(n_env):
            self.env_workers.append(
                self.create_worker(
                    worker_idx, self.step_queue, env_factory, run_options
                )
            )
            self.workers_alive += 1

    @staticmethod
    def create_worker(
        worker_id: int,
        step_queue: Queue,
        env_factory: Callable[[int, List[SideChannel]], BaseEnv],
        run_options: RunOptions,
    ) -> UnityEnvWorker:
        parent_conn, child_conn = Pipe()

        # Need to use cloudpickle for the env factory function since function objects aren't picklable
        # on Windows as of Python 3.6.
        pickled_env_factory = cloudpickle.dumps(env_factory)
        child_process = Process(
            target=worker,
            args=(
                child_conn,
                step_queue,
                pickled_env_factory,
                worker_id,
                run_options,
                logger.level,
            ),
        )
        child_process.start()
        return UnityEnvWorker(child_process, worker_id, parent_conn)

    def _queue_steps(self) -> None:
        for env_worker in self.env_workers:
            if not env_worker.waiting:
                env_action_info = self._take_step(env_worker.previous_step)
                env_worker.previous_all_action_info = env_action_info
                env_worker.send(EnvironmentCommand.STEP, env_action_info)
                env_worker.waiting = True

    def _restart_failed_workers(self, first_failure: EnvironmentResponse) -> None:
        if first_failure.cmd != EnvironmentCommand.ENV_EXITED:
            return
        # Drain the step queue to make sure all workers are paused and we have found all concurrent errors.
        # Pausing all training is needed since we need to reset all pending training steps as they could be corrupted.
        other_failures: Dict[int, Exception] = self._drain_step_queue()
        # Use Python 3.9+ dict union operator
        failures: Dict[int, Exception] = {
            first_failure.worker_id: first_failure.payload
        } | other_failures
        for worker_id, ex in failures.items():
            self._assert_worker_can_restart(worker_id, ex)
            logger.warning(f"Restarting worker[{worker_id}] after '{ex}'")
            self.recent_restart_timestamps[worker_id].append(datetime.datetime.now())
            self.restart_counts[worker_id] += 1
            self.env_workers[worker_id] = self.create_worker(
                worker_id, self.step_queue, self.env_factory, self.run_options
            )
        # The restarts were successful, clear all the existing training trajectories so we don't use corrupted or
        # outdated data.
        self.reset(self.env_parameters)

    def _drain_step_queue(self) -> Dict[int, Exception]:
        """
        Drains all steps out of the step queue and returns all exceptions from crashed workers.
        This will effectively pause all workers so that they won't do anything until _queue_steps is called.
        """
        all_failures = {}
        workers_still_pending = {w.worker_id for w in self.env_workers if w.waiting}
        deadline = datetime.datetime.now() + datetime.timedelta(minutes=1)
        while workers_still_pending and deadline > datetime.datetime.now():
            try:
                # Use timeout instead of busy polling
                step: EnvironmentResponse = self.step_queue.get(timeout=0.01)
                if step.cmd == EnvironmentCommand.ENV_EXITED:
                    workers_still_pending.add(step.worker_id)
                    all_failures[step.worker_id] = step.payload
                else:
                    workers_still_pending.remove(step.worker_id)
                    self.env_workers[step.worker_id].waiting = False
            except EmptyQueueException:
                continue
        if deadline < datetime.datetime.now():
            still_waiting = {w.worker_id for w in self.env_workers if w.waiting}
            raise TimeoutError(f"Workers {still_waiting} stuck in waiting state")
        return all_failures

    def _assert_worker_can_restart(self, worker_id: int, exception: Exception) -> None:
        """
        Checks if we can recover from an exception from a worker.
        If the restart limit is exceeded it will raise a UnityCommunicationException.
        If the exception is not recoverable it re-raises the exception.
        """
        if (
            isinstance(exception, UnityCommunicationException)
            or isinstance(exception, UnityTimeOutException)
            or isinstance(exception, UnityEnvironmentException)
            or isinstance(exception, UnityCommunicatorStoppedException)
        ):
            if self._worker_has_restart_quota(worker_id):
                return
            else:
                logger.error(
                    f"Worker {worker_id} exceeded the allowed number of restarts."
                )
                raise exception
        raise exception

    def _worker_has_restart_quota(self, worker_id: int) -> bool:
        self._drop_old_restart_timestamps(worker_id)
        max_lifetime_restarts = self.run_options.env_settings.max_lifetime_restarts
        max_limit_check = (
            max_lifetime_restarts == -1
            or self.restart_counts[worker_id] < max_lifetime_restarts
        )

        rate_limit_n = self.run_options.env_settings.restarts_rate_limit_n
        rate_limit_check = (
            rate_limit_n == -1
            or len(self.recent_restart_timestamps[worker_id]) < rate_limit_n
        )

        return rate_limit_check and max_limit_check

    def _drop_old_restart_timestamps(self, worker_id: int) -> None:
        """
        Drops environment restart timestamps that are outside of the current window.
        """

        def _filter(t: datetime.datetime) -> bool:
            return t > datetime.datetime.now() - datetime.timedelta(
                seconds=self.run_options.env_settings.restarts_rate_limit_period_s
            )

        self.recent_restart_timestamps[worker_id] = list(
            filter(_filter, self.recent_restart_timestamps[worker_id])
        )

    def _step(self) -> List[EnvironmentStep]:
        # Queue steps for any workers which aren't in the "waiting" state.
        self._queue_steps()

        worker_steps: List[EnvironmentResponse] = []
        step_workers: Set[int] = set()
        # Poll the step queue for completed steps from environment workers until we retrieve
        # 1 or more, which we will then return as StepInfos
        while len(worker_steps) < 1:
            try:
                # Use timeout instead of busy polling to reduce CPU usage
                step: EnvironmentResponse = self.step_queue.get(timeout=0.001)
                if step.cmd == EnvironmentCommand.ENV_EXITED:
                    # If even one env exits try to restart all envs that failed.
                    self._restart_failed_workers(step)
                    # Clear state and restart this function.
                    worker_steps.clear()
                    step_workers.clear()
                    self._queue_steps()
                elif step.worker_id not in step_workers:
                    self.env_workers[step.worker_id].waiting = False
                    worker_steps.append(step)
                    step_workers.add(step.worker_id)
            except EmptyQueueException:
                continue
        step_infos = self._postprocess_steps(worker_steps)
        return step_infos

    def _reset_env(self, config: Optional[Dict] = None) -> List[EnvironmentStep]:
        while any(ew.waiting for ew in self.env_workers):
            if not self.step_queue.empty():
                step = self.step_queue.get_nowait()
                self.env_workers[step.worker_id].waiting = False
        # Send config to environment
        self.set_env_parameters(config)
        # First enqueue reset commands for all workers so that they reset in parallel
        for ew in self.env_workers:
            ew.send(EnvironmentCommand.RESET, config)
        # Next (synchronously) collect the reset observations from each worker in sequence
        for ew in self.env_workers:
            ew.previous_step = EnvironmentStep(ew.recv().payload, ew.worker_id, {}, {})
        return list(map(lambda ew: ew.previous_step, self.env_workers))

    def set_env_parameters(self, config: Dict = None) -> None:
        """
        Sends environment parameter settings to C# via the
        EnvironmentParametersSidehannel for each worker.
        :param config: Dict of environment parameter keys and values
        """
        self.env_parameters = config
        for ew in self.env_workers:
            ew.send(EnvironmentCommand.ENVIRONMENT_PARAMETERS, config)

    def on_training_started(
        self, behavior_name: str, trainer_settings: TrainerSettings
    ) -> None:
        """
        Handle traing starting for a new behavior type. Generally nothing is necessary here.
        :param behavior_name:
        :param trainer_settings:
        :return:
        """
        for ew in self.env_workers:
            ew.send(
                EnvironmentCommand.TRAINING_STARTED, (behavior_name, trainer_settings)
            )

    @property
    def training_behaviors(self) -> Dict[BehaviorName, BehaviorSpec]:
        result: Dict[BehaviorName, BehaviorSpec] = {}
        for worker in self.env_workers:
            worker.send(EnvironmentCommand.BEHAVIOR_SPECS)
            result.update(worker.recv().payload)
        return result

    def close(self) -> None:
        logger.debug("SubprocessEnvManager closing.")
        for env_worker in self.env_workers:
            env_worker.request_close()
        # Pull messages out of the queue until every worker has CLOSED or we time out.
        deadline = time.time() + WORKER_SHUTDOWN_TIMEOUT_S
        while self.workers_alive > 0 and time.time() < deadline:
            try:
                step: EnvironmentResponse = self.step_queue.get_nowait()
                env_worker = self.env_workers[step.worker_id]
                if step.cmd == EnvironmentCommand.CLOSED and not env_worker.closed:
                    env_worker.closed = True
                    self.workers_alive -= 1
                # Discard all other messages.
            except EmptyQueueException:
                pass
        self.step_queue.close()
        # Sanity check to kill zombie workers and report an issue if they occur.
        if self.workers_alive > 0:
            logger.error("SubprocessEnvManager had workers that didn't signal shutdown")
            for env_worker in self.env_workers:
                if not env_worker.closed and env_worker.process.is_alive():
                    env_worker.process.terminate()
                    logger.error(
                        "A SubprocessEnvManager worker did not shut down correctly so it was forcefully terminated."
                    )
        self.step_queue.join_thread()

    def _postprocess_steps(
        self, env_steps: List[EnvironmentResponse]
    ) -> List[EnvironmentStep]:
        step_infos = []
        timer_nodes = []
        for step in env_steps:
            payload: StepResponse = step.payload
            env_worker = self.env_workers[step.worker_id]
            new_step = EnvironmentStep(
                payload.all_step_result,
                step.worker_id,
                env_worker.previous_all_action_info,
                payload.environment_stats,
            )
            step_infos.append(new_step)
            env_worker.previous_step = new_step

            if payload.timer_root:
                timer_nodes.append(payload.timer_root)

        if timer_nodes:
            with hierarchical_timer("workers") as main_timer_node:
                for worker_timer_node in timer_nodes:
                    main_timer_node.merge(
                        worker_timer_node, root_name="worker_root", is_parallel=True
                    )

        return step_infos

    @timed
    def _take_step(self, last_step: EnvironmentStep) -> Dict[BehaviorName, ActionInfo]:
        all_action_info: Dict[str, ActionInfo] = {}
        for brain_name, step_tuple in last_step.current_all_step_result.items():
            if brain_name in self.policies:
                all_action_info[brain_name] = self.policies[brain_name].get_action(
                    step_tuple[0], last_step.worker_id
                )
        return all_action_info
