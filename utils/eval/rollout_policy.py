from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
import os
from pathlib import Path
import time
from typing import Any
import uuid

import gymnasium as gym
import numpy as np
from tqdm import tqdm
import tyro

from utils.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from utils.policy import BasePolicy
from utils.policy.policy_core import HelmSimPolicyWrapper
from utils.policy.helm_policy import HelmPolicy
from utils.determinism import seed_everything


@contextmanager
def _skip_unreadable_proc_cpuinfo_for_pyopengl():
    proc_cpuinfo = "/proc/cpuinfo"
    original_exists = os.path.exists

    def safe_exists(path):
        if path == proc_cpuinfo:
            try:
                with open(proc_cpuinfo, "r") as stream:
                    stream.read()
            except OSError:
                return False
        return original_exists(path)

    os.path.exists = safe_exists
    try:
        yield
    finally:
        os.path.exists = original_exists


@dataclass
class VideoConfig:
    video_dir: str | None = None
    steps_per_render: int = 2
    max_episode_steps: int = 720
    fps: int = 20
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1
    overlay_text: bool = True


@dataclass
class MultiStepConfig:
    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 8
    max_episode_steps: int = 720
    terminate_on_success: bool = True


@dataclass
class WrapperConfigs:
    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


def create_robocasa_env(env_name: str) -> gym.Env:
    if not env_name.startswith("gr1_unified/"):
        raise ValueError("HELM simulation supports only gr1_unified RoboCasa environments")
    if os.environ.get("MUJOCO_GL", "").lower() == "egl":
        os.environ.pop("PYOPENGL_PLATFORM", None)
    with _skip_unreadable_proc_cpuinfo_for_pyopengl():
        import robocasa  # noqa: F401
        import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

    return gym.make(env_name, enable_render=True)


def create_eval_env(env_name: str, wrapper_configs: WrapperConfigs) -> gym.Env:
    env = create_robocasa_env(env_name)
    if wrapper_configs.video.video_dir is not None:
        from utils.eval.sim.wrapper.video_recording_wrapper import (
            VideoRecorder,
            VideoRecordingWrapper,
        )

        video_recorder = VideoRecorder.create_h264(
            fps=wrapper_configs.video.fps,
            codec=wrapper_configs.video.codec,
            input_pix_fmt=wrapper_configs.video.input_pix_fmt,
            crf=wrapper_configs.video.crf,
            thread_type=wrapper_configs.video.thread_type,
            thread_count=wrapper_configs.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(wrapper_configs.video.video_dir),
            steps_per_render=wrapper_configs.video.steps_per_render,
            max_episode_steps=wrapper_configs.video.max_episode_steps,
            overlay_text=wrapper_configs.video.overlay_text,
        )
    return MultiStepWrapper(
        env,
        video_delta_indices=wrapper_configs.multistep.video_delta_indices,
        state_delta_indices=wrapper_configs.multistep.state_delta_indices,
        n_action_steps=wrapper_configs.multistep.n_action_steps,
        max_episode_steps=wrapper_configs.multistep.max_episode_steps,
        terminate_on_success=wrapper_configs.multistep.terminate_on_success,
    )


class _RobustAsyncVectorEnv(gym.vector.AsyncVectorEnv):
    def _add_info(self, infos, info, env_num):
        for key, value in info.items():
            if key not in infos:
                infos[key] = [None] * self.num_envs
                infos[f"_{key}"] = np.zeros(self.num_envs, dtype=bool)
            if isinstance(infos[key], np.ndarray):
                try:
                    infos[key][env_num] = value
                except (ValueError, TypeError):
                    values = list(infos[key])
                    values[env_num] = value
                    infos[key] = values
            else:
                infos[key][env_num] = value
            infos[f"_{key}"][env_num] = True
        return infos


def _as_success(value: Any) -> bool:
    if isinstance(value, (list, np.ndarray)):
        return bool(np.any(value))
    if isinstance(value, (bool, int, np.bool_)):
        return bool(value)
    raise TypeError(f"Unsupported success value: {type(value)}")


def run_rollouts(
    env_name: str,
    policy: BasePolicy,
    wrapper_configs: WrapperConfigs,
    n_episodes: int,
    n_envs: int,
    seed: int | None,
) -> tuple[str, list[bool], dict[str, list[Any]]]:
    start_time = time.time()
    n_episodes = max(n_episodes, n_envs)
    env_fns = [partial(create_eval_env, env_name, wrapper_configs) for _ in range(n_envs)]
    if n_envs == 1:
        env = gym.vector.SyncVectorEnv(env_fns)
    else:
        env = _RobustAsyncVectorEnv(env_fns, shared_memory=False, context="spawn")

    reset_seeds = [seed + index for index in range(n_envs)] if seed is not None else None
    observations, _ = env.reset(seed=reset_seeds)
    policy.reset()
    successes: list[bool] = []
    episode_infos: defaultdict[str, list[Any]] = defaultdict(list)
    rewards = [0.0] * n_envs
    lengths = [0] * n_envs
    active_success = [False] * n_envs

    progress = tqdm(total=n_episodes, desc="Episodes")
    try:
        while len(successes) < n_episodes:
            actions, _ = policy.get_action(observations)
            observations, step_rewards, terminated, truncated, infos = env.step(actions)
            for index in range(n_envs):
                if "success" in infos:
                    active_success[index] |= _as_success(infos["success"][index])
                final_info = infos.get("final_info", [None] * n_envs)[index]
                if final_info is not None and "success" in final_info:
                    active_success[index] |= _as_success(final_info["success"])
                rewards[index] += float(step_rewards[index])
                lengths[index] += 1
                if not (terminated[index] or truncated[index]):
                    continue
                if len(successes) < n_episodes:
                    successes.append(active_success[index])
                    episode_infos["episode_rewards"].append(rewards[index])
                    episode_infos["episode_lengths"].append(lengths[index])
                    progress.update(1)
                active_success[index] = False
                rewards[index] = 0.0
                lengths[index] = 0
    finally:
        progress.close()
        env.close()

    print(f"Collected {len(successes)} episodes in {time.time() - start_time:.1f}s")
    return env_name, successes, dict(episode_infos)


@dataclass
class RolloutConfig:
    model_path: str
    pretrain_path: str = "data/helm_pretrain"
    env_name: str = (
        "gr1_unified/"
        "PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env"
    )
    n_episodes: int = 20
    n_envs: int = 5
    n_action_steps: int = 8
    max_episode_steps: int = 720
    video_dir: str | None = None
    merge_lora: bool = True
    seed: int | None = None


def main(config: RolloutConfig) -> None:
    seed = seed_everything(config.seed)
    video_dir = config.video_dir or f"/tmp/helm_eval_{uuid.uuid4()}"
    policy = HelmSimPolicyWrapper(
        HelmPolicy(
            embodiment_tag="ROBOCASA_GR1_TABLETOP",
            model_path=config.model_path,
            device=0,
            vlm_model_path=f"{config.pretrain_path}/vlm",
            wam_model_path=f"{config.pretrain_path}/wam",
            wam_text_embedding_cache=f"{config.pretrain_path}/text_embeddings.pt",
            merge_lora=config.merge_lora,
        )
    )
    result = run_rollouts(
        env_name=config.env_name,
        policy=policy,
        wrapper_configs=WrapperConfigs(
            video=VideoConfig(video_dir=video_dir, max_episode_steps=config.max_episode_steps),
            multistep=MultiStepConfig(
                n_action_steps=config.n_action_steps,
                max_episode_steps=config.max_episode_steps,
            ),
        ),
        n_episodes=config.n_episodes,
        n_envs=config.n_envs,
        seed=seed,
    )
    print(f"Video saved to: {video_dir}")
    print(f"success rate: {np.mean(result[1]):.6f}")


if __name__ == "__main__":
    main(tyro.cli(RolloutConfig))
