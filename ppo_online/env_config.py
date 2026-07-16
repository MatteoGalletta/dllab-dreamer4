from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

PUSHT_ENV_ID = "swm/PushT-v1"
DEFAULT_PUSHT_ENV_ID = PUSHT_ENV_ID
PUSHT_RENDER_SHAPE = (96, 96, 3)
PUSHT_FIXED_TARGET_POSE = np.array([256.0, 256.0, np.pi / 4], dtype=np.float64)
PUSHT_WORKSPACE_LOW = np.array([0.0, 0.0], dtype=np.float64)
PUSHT_WORKSPACE_HIGH = np.array([512.0, 512.0], dtype=np.float64)


def _try_register_optional_envs() -> None:
    # Some env packages register their Gym environments only as an import side effect.
    _try_patch_pymunk_for_swm()
    try:
        import stable_worldmodel  # noqa: F401
    except Exception:
        pass


def _try_patch_pymunk_for_swm() -> None:
    try:
        import pymunk
    except Exception:
        return

    if hasattr(pymunk.Space, "on_collision"):
        return
    if not hasattr(pymunk.Space, "add_collision_handler"):
        return

    def on_collision(self, collision_type_a, collision_type_b, begin=None, pre_solve=None, post_solve=None, separate=None):
        handler = self.add_collision_handler(collision_type_a, collision_type_b)
        if begin is not None:
            handler.begin = begin
        if pre_solve is not None:
            handler.pre_solve = pre_solve
        if post_solve is not None:
            handler.post_solve = post_solve
        if separate is not None:
            handler.separate = separate
        return handler

    pymunk.Space.on_collision = on_collision


def ensure_swm_compat() -> None:
    _try_patch_pymunk_for_swm()


def _wrap_angle(angle):
    return float(angle % (2 * np.pi))


def _angle_distance(angle_a, angle_b):
    diff = abs(float(angle_a) - float(angle_b))
    return min(diff, 2 * np.pi - diff)


def _rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _polygon_area_centroid(vertices):
    verts = np.asarray(vertices, dtype=np.float64)
    x, y = verts[:, 0], verts[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = 0.5 * float(cross.sum())
    if abs(area) < 1e-9:
        return 0.0, verts.mean(axis=0)
    cx = float(((x + x_next) * cross).sum()) / (6.0 * area)
    cy = float(((y + y_next) * cross).sum()) / (6.0 * area)
    return abs(area), np.array([cx, cy], dtype=np.float64)


def _block_local_centroid(unwrapped):
    total_area = 0.0
    weighted = np.zeros(2, dtype=np.float64)
    for shape in unwrapped.block.shapes:
        get_vertices = getattr(shape, "get_vertices", None)
        if get_vertices is not None:
            area, centroid = _polygon_area_centroid([tuple(v) for v in get_vertices()])
        else:
            radius = float(getattr(shape, "radius", 0.0))
            area = float(np.pi * radius * radius)
            centroid = np.asarray(tuple(getattr(shape, "offset", (0.0, 0.0))), dtype=np.float64)
        total_area += area
        weighted += area * centroid
    return weighted / total_area if total_area > 0 else np.zeros(2)


def green_t_center(env):
    unwrapped = env.unwrapped
    goal_pose = np.asarray(unwrapped.goal_pose, dtype=np.float64)
    return goal_pose[:2] + _rotation_matrix(float(goal_pose[2])) @ _block_local_centroid(unwrapped)


def block_center(env):
    unwrapped = env.unwrapped
    state = np.asarray(unwrapped._get_obs(), dtype=np.float64)
    block_pos, block_angle = state[2:4], float(state[4])
    return block_pos + _rotation_matrix(block_angle) @ _block_local_centroid(unwrapped)


def _success_from_info(info, terminated):
    for key in ("success", "is_success", "task_success"):
        if key in info:
            return float(info[key]) > 0.5
    return bool(terminated)


class PushTGoalPoseFromStateWrapper(gym.Wrapper):
    """Keep the rendered goal pose aligned with the goal state."""

    def _sync_goal_pose(self, info=None):
        env = self.unwrapped
        goal_state = getattr(env, "goal_state", None)
        if goal_state is None:
            return info

        goal_state = np.asarray(goal_state)
        if goal_state.shape[0] < 5:
            return info

        env.goal_pose = goal_state[2:5].copy()
        if info is not None:
            info = dict(info)
            info["goal_pose"] = env.goal_pose
        return info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return observation, self._sync_goal_pose(info)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, self._sync_goal_pose(info)


class PushTAlignSampledGoalToFixedTargetWrapper(gym.Wrapper):
    """Rigidly align each sampled PushT task to a fixed block target pose."""

    def __init__(
        self,
        env,
        target_pose=PUSHT_FIXED_TARGET_POSE,
        block_success=True,
        block_position_threshold=20.0,
        block_angle_threshold=np.pi / 9,
        max_reset_attempts=100,
        workspace_low=PUSHT_WORKSPACE_LOW,
        workspace_high=PUSHT_WORKSPACE_HIGH,
        agent_block_coef=0.0,
    ):
        super().__init__(env)
        self.target_pose = np.asarray(target_pose, dtype=np.float64)
        self.block_success = bool(block_success)
        self.agent_block_coef = float(agent_block_coef)
        self.block_position_threshold = float(block_position_threshold)
        self.block_angle_threshold = float(block_angle_threshold)
        self.max_reset_attempts = max(1, int(max_reset_attempts))
        self.workspace_low = np.asarray(workspace_low, dtype=np.float64)
        self.workspace_high = np.asarray(workspace_high, dtype=np.float64)

    def _transform_state(self, state, sampled_goal_pose):
        state = np.asarray(state, dtype=np.float64).copy()
        dtheta = self.target_pose[2] - sampled_goal_pose[2]
        rotation = _rotation_matrix(dtheta)
        state[:2] = self.target_pose[:2] + rotation @ (state[:2] - sampled_goal_pose[:2])
        state[2:4] = self.target_pose[:2] + rotation @ (state[2:4] - sampled_goal_pose[:2])
        state[4] = _wrap_angle(state[4] + dtheta)
        if state.shape[0] >= 7:
            state[-2:] = rotation @ state[-2:]
        return state

    def _is_valid_state(self, state):
        agent_xy = state[:2]
        block_xy = state[2:4]
        return (
            np.all(agent_xy >= self.workspace_low)
            and np.all(agent_xy <= self.workspace_high)
            and np.all(block_xy >= self.workspace_low)
            and np.all(block_xy <= self.workspace_high)
        )

    def _format_observation(self):
        env = self.unwrapped
        state = env._get_obs()
        proprio = np.concatenate((state[:2], state[-2:]))
        return {"proprio": proprio, "state": state}

    def _refresh_goal_image(self, current_state):
        env = self.unwrapped
        if not hasattr(env, "_goal"):
            return
        env._set_state(env.goal_state)
        env._goal = env.render()
        env._set_state(current_state)

    def _block_metrics(self):
        env = self.unwrapped
        state = env._get_obs()
        block_pose = state[2:5]
        goal_pose = np.asarray(env.goal_state[2:5], dtype=np.float64)
        pos_dist = float(np.linalg.norm(goal_pose[:2] - block_pose[:2]))
        angle_dist = _angle_distance(goal_pose[2], block_pose[2])
        state_dist = float(np.linalg.norm([pos_dist, angle_dist]))
        success = pos_dist < self.block_position_threshold and angle_dist < self.block_angle_threshold
        return success, pos_dist, angle_dist, state_dist

    def _augment_info(self):
        env = self.unwrapped
        info = dict(env._get_info())
        info["goal_pose"] = np.asarray(env.goal_pose).copy()
        info["goal_state"] = np.asarray(env.goal_state).copy()
        state = env._get_obs()
        info["agent_block_dist"] = float(np.linalg.norm(state[:2] - state[2:4]))
        if self.block_success:
            success, pos_dist, angle_dist, state_dist = self._block_metrics()
            info.update(
                {
                    "success": float(success),
                    "block_success": float(success),
                    "block_pos_dist": pos_dist,
                    "block_angle_dist": angle_dist,
                    "block_state_dist": state_dist,
                }
            )
        return info

    def _apply_alignment(self):
        env = self.unwrapped
        current_state = env._get_obs()
        sampled_goal_state = np.asarray(env.goal_state, dtype=np.float64).copy()
        sampled_goal_pose = sampled_goal_state[2:5].copy()
        aligned_state = self._transform_state(current_state, sampled_goal_pose)
        aligned_goal_state = self._transform_state(sampled_goal_state, sampled_goal_pose)
        aligned_goal_state[2:5] = self.target_pose.copy()
        if not self._is_valid_state(aligned_state):
            return None, None
        env._set_goal_state(aligned_goal_state)
        env.goal_pose = self.target_pose.copy()
        env._set_state(aligned_state)
        self._refresh_goal_image(aligned_state)
        return self._format_observation(), self._augment_info()

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        base_kwargs = dict(kwargs)
        for attempt in range(self.max_reset_attempts):
            reset_kwargs = dict(base_kwargs)
            if seed is not None and attempt > 0:
                reset_kwargs["seed"] = int(seed) + attempt
            self.env.reset(**reset_kwargs)
            observation, info = self._apply_alignment()
            if observation is not None:
                return observation, info
        raise RuntimeError(
            "Could not sample a valid fixed-target PushT episode after "
            f"{self.max_reset_attempts} reset attempts."
        )

    def step(self, action):
        observation, reward, terminated, truncated, _ = self.env.step(action)
        self.unwrapped.goal_pose = self.target_pose.copy()
        info = self._augment_info()
        if self.block_success:
            terminated = bool(info["block_success"])
            reward = -float(info["block_state_dist"])
            if self.agent_block_coef:
                reward -= self.agent_block_coef * info["agent_block_dist"]
        return observation, reward, terminated, truncated, info


class PushTRenderObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, image_shape=PUSHT_RENDER_SHAPE):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)

    def observation(self, observation):
        del observation
        return np.asarray(self.env.render(), dtype=np.uint8)


class PushTBlockStartNearGoalWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        radius=50.0,
        bounds_margin=20.0,
        min_agent_clearance=0.0,
        max_sample_attempts=100,
        workspace_low=PUSHT_WORKSPACE_LOW,
        workspace_high=PUSHT_WORKSPACE_HIGH,
    ):
        super().__init__(env)
        self.radius = float(radius)
        self.bounds_margin = float(bounds_margin)
        self.min_agent_clearance = float(min_agent_clearance)
        self.max_sample_attempts = max(1, int(max_sample_attempts))
        self.workspace_low = np.asarray(workspace_low, dtype=np.float64)
        self.workspace_high = np.asarray(workspace_high, dtype=np.float64)
        self._low = self.workspace_low + self.bounds_margin
        self._high = self.workspace_high - self.bounds_margin
        self._rng = np.random.default_rng()

    def _sample_block_pos(self, goal_center, agent_xy, block_angle, local_centroid):
        rot = _rotation_matrix(block_angle)
        best = None
        for _ in range(self.max_sample_attempts):
            radius = self.radius * np.sqrt(self._rng.random())
            theta = self._rng.uniform(0.0, 2.0 * np.pi)
            target_centroid = goal_center + radius * np.array([np.cos(theta), np.sin(theta)])
            block_pos = np.clip(target_centroid - rot @ local_centroid, self._low, self._high)
            best = block_pos
            resulting_centroid = block_pos + rot @ local_centroid
            if self.min_agent_clearance <= 0.0 or np.linalg.norm(resulting_centroid - agent_xy) >= self.min_agent_clearance:
                return block_pos
        return best

    def _reposition_block(self, info):
        env = self.unwrapped
        state = np.asarray(env._get_obs(), dtype=np.float64)
        goal_center = green_t_center(env)
        local_centroid = _block_local_centroid(env)
        new_block_pos = self._sample_block_pos(goal_center, state[:2], float(state[4]), local_centroid)
        new_state = state.copy()
        new_state[2:4] = new_block_pos
        env._set_state(new_state)
        state = np.asarray(env._get_obs(), dtype=np.float64)
        observation = {"proprio": np.concatenate((state[:2], state[-2:])), "state": state}
        info = dict(info)
        info["green_t_center"] = goal_center
        info["block_pose"] = np.array(list(state[2:4]) + [state[4]])
        info["block_goal_dist"] = float(np.linalg.norm(block_center(env) - goal_center))
        if "agent_block_dist" in info:
            info["agent_block_dist"] = float(np.linalg.norm(state[:2] - state[2:4]))
        return observation, info

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        _, info = self.env.reset(**kwargs)
        return self._reposition_block(info)


class PushTRewardModeWrapper(gym.Wrapper):
    def __init__(self, env, reward_mode="dense", success_reward=1.0, failure_reward=0.0):
        super().__init__(env)
        if reward_mode not in ("dense", "sparse"):
            raise ValueError("reward_mode must be 'dense' or 'sparse'")
        self.reward_mode = reward_mode
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.reward_mode == "sparse":
            success = _success_from_info(info, terminated)
            reward = self.success_reward if success else self.failure_reward
        return observation, float(reward), terminated, truncated, info


def resolve_pusht_env_id(preferred_env_id: str = DEFAULT_PUSHT_ENV_ID) -> str:
    if preferred_env_id != PUSHT_ENV_ID:
        raise ValueError(f"Only the configured PushT environment '{PUSHT_ENV_ID}' is supported, got '{preferred_env_id}'.")

    _try_register_optional_envs()
    try:
        gym.spec(PUSHT_ENV_ID)
    except Exception as error:
        raise gym.error.NameNotFound(
            f"Required PushT environment '{PUSHT_ENV_ID}' is not registered. "
            "Install the stable-worldmodel environment package."
        ) from error
    return PUSHT_ENV_ID


def make_pusht_env_kwargs(
    env_id: str,
    render_mode: str = "rgb_array",
    image_height: int | None = None,
    image_width: int | None = None,
) -> dict:
    if env_id != PUSHT_ENV_ID:
        raise ValueError(f"Only the configured PushT environment '{PUSHT_ENV_ID}' is supported, got '{env_id}'.")

    resolution = int(image_height or image_width or 224)
    return {
        "render_mode": render_mode,
        "resolution": resolution,
        "relative": False,
    }


def make_pusht_env(
    *,
    env_id: str = DEFAULT_PUSHT_ENV_ID,
    render_mode: str = "rgb_array",
    render_obs: bool = False,
    sync_goal_pose: bool = True,
    align_sampled_goal_to_fixed_target: bool = False,
    fixed_target_pose=PUSHT_FIXED_TARGET_POSE,
    fixed_target_block_success: bool = True,
    fixed_target_max_reset_attempts: int = 100,
    fixed_target_agent_block_coef: float = 0.0,
    block_start_near_goal: bool = False,
    block_start_radius: float = 50.0,
    block_start_min_agent_clearance: float = 0.0,
    reward_mode: str = "dense",
    image_height: int | None = None,
    image_width: int | None = None,
    **kwargs,
):
    resolved_env_id = resolve_pusht_env_id(env_id)
    env_kwargs = make_pusht_env_kwargs(
        resolved_env_id,
        render_mode=render_mode,
        image_height=image_height,
        image_width=image_width,
    )
    env_kwargs.update(kwargs)
    # The upstream SWM PushT env can emit observations that violate its own
    # declared Gym space slightly, which triggers noisy passive checker
    # warnings during BC/PPO rollouts even though our wrappers handle the data.
    env = gym.make(resolved_env_id, disable_env_checker=True, **env_kwargs)
    if align_sampled_goal_to_fixed_target:
        env = PushTAlignSampledGoalToFixedTargetWrapper(
            env,
            target_pose=fixed_target_pose,
            block_success=fixed_target_block_success,
            max_reset_attempts=fixed_target_max_reset_attempts,
            agent_block_coef=fixed_target_agent_block_coef,
        )
    if sync_goal_pose:
        env = PushTGoalPoseFromStateWrapper(env)
    if block_start_near_goal:
        env = PushTBlockStartNearGoalWrapper(
            env,
            radius=block_start_radius,
            min_agent_clearance=block_start_min_agent_clearance,
        )
    if reward_mode != "dense":
        env = PushTRewardModeWrapper(env, reward_mode=reward_mode)
    if render_obs:
        env = PushTRenderObservationWrapper(env)
    return env
