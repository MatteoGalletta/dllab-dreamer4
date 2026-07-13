from __future__ import annotations

import gymnasium as gym

DEFAULT_PUSHT_ENV_ID = "swm/PushT-v1"
FALLBACK_PUSHT_ENV_IDS = (
    "gym_pusht/PushT-v0",
    "PushT-v0",
)


def _try_register_optional_envs(preferred_env_id: str) -> None:
    # Some env packages register their Gym environments only as an import side effect.
    if preferred_env_id.startswith("swm/"):
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


def resolve_pusht_env_id(preferred_env_id: str = DEFAULT_PUSHT_ENV_ID) -> str:
    _try_register_optional_envs(preferred_env_id)
    candidate_ids = (preferred_env_id, *FALLBACK_PUSHT_ENV_IDS)
    for env_id in candidate_ids:
        try:
            gym.spec(env_id)
            if env_id != preferred_env_id:
                print(
                    f"Preferred PushT env '{preferred_env_id}' is unavailable. "
                    f"Falling back to installed env '{env_id}'."
                )
            return env_id
        except Exception:
            continue

    raise gym.error.NameNotFound(
        "No supported PushT environment is registered. Tried: "
        + ", ".join(candidate_ids)
        + ". Install the 'swm' environment package or the legacy 'gym_pusht' package."
    )


def make_pusht_env_kwargs(
    env_id: str,
    render_mode: str = "rgb_array",
    image_height: int | None = None,
    image_width: int | None = None,
) -> dict:
    if env_id.startswith("swm/"):
        resolution = int(image_height or image_width or 224)
        return {
            "render_mode": render_mode,
            "resolution": resolution,
            "relative": False,
        }

    kwargs = {
        "obs_type": "state",
        "render_mode": render_mode,
    }
    if image_height is not None:
        kwargs["observation_height"] = int(image_height)
    if image_width is not None:
        kwargs["observation_width"] = int(image_width)
    return kwargs
