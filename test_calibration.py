#!/usr/bin/env python
"""
Calibration Test Script for LeRobot

This script records test episodes locally (without cameras, joints only) and replays them
to test calibration accuracy by comparing recorded vs replayed joint positions.

Works with pip-installed lerobot package - uses only public API.

Usage:
    python test_calibration.py \
        --robot.type=so100_follower \
        --robot.port=/dev/tty.usbmodem58760431541 \
        --teleop.type=so100_leader \
        --teleop.port=/dev/tty.usbmodem58760431551 \
        --dataset.repo_id=calibration_test \
        --dataset.root=./local_datasets \
        --dataset.num_episodes=3 \
        --dataset.episode_time_s=30
"""

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch

# All imports use public lerobot API (available in pip package)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say

# Import robot modules to register their types with RobotConfig
# This ensures robot types are available when the parser validates choices
try:
    from lerobot.robots import so100_follower, so101_follower  # noqa: F401
except ImportError:
    pass  # Robot modules may not all be available in all installations

# Optional import for third-party plugins (may not be available in all versions)
try:
    from lerobot.utils.import_utils import register_third_party_plugins
except ImportError:
    # Fallback if function doesn't exist in pip package
    def register_third_party_plugins():
        pass  # No-op if not available


# Calibration tolerance thresholds (adjust based on your robot)
JOINT_POSITION_TOLERANCE_RAD = 0.05  # ~2.9 degrees
MAX_JOINT_ERROR_TOLERANCE_RAD = 0.1  # ~5.7 degrees


def compare_observations(
    recorded_obs: dict[str, Any],
    replayed_obs: dict[str, Any],
    tolerance: float = JOINT_POSITION_TOLERANCE_RAD,
) -> dict[str, Any]:
    """Compare recorded and replayed observations.
    
    Args:
        recorded_obs: Observation from recorded dataset
        replayed_obs: Observation from robot during replay
        tolerance: Maximum allowed error in radians
        
    Returns:
        Dictionary with comparison results
    """
    errors = {}
    
    # Compare state (joint positions)
    if "observation.state" in recorded_obs and "observation.state" in replayed_obs:
        recorded_state = recorded_obs["observation.state"]
        replayed_state = replayed_obs["observation.state"]
        
        # Convert to numpy if needed
        if isinstance(recorded_state, torch.Tensor):
            recorded_state = recorded_state.numpy()
        if isinstance(replayed_state, torch.Tensor):
            replayed_state = replayed_state.numpy()
        
        # Ensure same shape
        if recorded_state.ndim > 1:
            recorded_state = recorded_state.flatten()
        if replayed_state.ndim > 1:
            replayed_state = replayed_state.flatten()
        
        state_error = np.abs(recorded_state - replayed_state)
        errors["state"] = {
            "max_error": np.max(state_error),
            "mean_error": np.mean(state_error),
            "errors": state_error,
            "within_tolerance": np.max(state_error) < tolerance,
        }
    
    return errors


@dataclass
class DatasetTestConfig:
    """Configuration for dataset recording and testing."""
    # Dataset identifier (used as directory name)
    repo_id: str = "calibration_test"
    # Root directory where the dataset will be stored
    root: str | Path | None = None
    # Number of episodes to record
    num_episodes: int = 3
    # Duration of each episode in seconds
    episode_time_s: float = 30.0
    # Frames per second
    fps: int = 30
    # Task description
    task: str = "Calibration test trajectory"
    # Episode indices to test (None = all episodes)
    test_episodes: list[int] | None = None
    # Skip recording, only replay existing dataset
    skip_record: bool = False


@dataclass
class CalibrationTestConfig:
    """Configuration for calibration testing."""
    robot: RobotConfig
    teleop: TeleoperatorConfig | None = None
    dataset: DatasetTestConfig = None
    # Joint error tolerance in radians
    tolerance: float = JOINT_POSITION_TOLERANCE_RAD
    # Use vocal synthesis to read events
    play_sounds: bool = True


def record_test_episode(
    robot_config: RobotConfig,
    teleop_config: TeleoperatorConfig | None,
    dataset_config: DatasetTestConfig,
) -> LeRobotDataset:
    """Record test episodes for calibration testing.
    
    Args:
        robot_config: Robot configuration
        teleop_config: Teleoperator configuration (None if using policy)
        dataset_config: Dataset configuration
        
    Returns:
        Recorded dataset
    """
    print(f"\n{'='*60}")
    print("STEP 1: Recording Test Episodes")
    print(f"{'='*60}")
    
    # Initialize robot and teleop
    robot = make_robot_from_config(robot_config)
    teleop = make_teleoperator_from_config(teleop_config) if teleop_config is not None else None
    
    # Setup processors
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    
    # Create dataset features (joints only, no cameras)
    action_features = hw_to_dataset_features(robot.action_features, ACTION, use_video=False)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False)
    
    # Combine features using pipeline aggregation
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=False,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=False,
        ),
    )
    
    # Create dataset locally
    dataset = LeRobotDataset.create(
        repo_id=dataset_config.repo_id,
        fps=dataset_config.fps,
        root=dataset_config.root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=False,  # No cameras, so no videos
        image_writer_processes=0,
        image_writer_threads=0,
    )
    
    # Connect robot and teleop
    robot.connect()
    if teleop is not None:
        teleop.connect()
    
    if not robot.is_connected:
        raise ValueError("Robot is not connected!")
    if teleop is not None and not teleop.is_connected:
        raise ValueError("Teleoperator is not connected!")
    
    print(f"Recording {dataset_config.num_episodes} episode(s) of {dataset_config.episode_time_s}s each...")
    print(f"Dataset will be saved to: {dataset.root}")
    print(f"Task: {dataset_config.task}")
    
    # Recording loop
    for episode_idx in range(dataset_config.num_episodes):
        print(f"\nRecording episode {episode_idx + 1}/{dataset_config.num_episodes}...")
        log_say(f"Recording episode {episode_idx + 1}", dataset_config.play_sounds, blocking=False)
        
        start_time = time.perf_counter()
        frame_count = 0
        
        while (time.perf_counter() - start_time) < dataset_config.episode_time_s:
            frame_start = time.perf_counter()
            
            # Get robot observation
            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)
            
            # Get action from teleop
            if teleop is not None:
                action = teleop.get_action()
                action_processed = teleop_action_processor((action, obs))
            else:
                # If no teleop, use neutral/zero action (for testing)
                action_processed = {name: 0.0 for name in robot.action_features.keys()}
            
            # Process action for robot
            robot_action = robot_action_processor((action_processed, obs))
            
            # Send action to robot
            _ = robot.send_action(robot_action)
            
            # Build and save frame
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset.features, action_processed, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": dataset_config.task}
            dataset.add_frame(frame)
            
            frame_count += 1
            
            # Maintain FPS
            dt = time.perf_counter() - frame_start
            precise_sleep(1.0 / dataset_config.fps - dt)
        
        # Save episode
        dataset.save_episode()
        print(f"  Episode {episode_idx + 1} recorded: {frame_count} frames")
    
    # Finalize dataset
    dataset.finalize()
    
    # Disconnect
    robot.disconnect()
    if teleop is not None:
        teleop.disconnect()
    
    print(f"\n✓ Recording complete! Dataset saved to: {dataset.root}")
    print(f"  Total episodes: {dataset.num_episodes}")
    print(f"  Total frames: {dataset.num_frames}")
    
    return dataset


def replay_and_test_calibration(
    dataset: LeRobotDataset,
    robot_config: RobotConfig,
    episode_indices: list[int] | None = None,
    tolerance: float = JOINT_POSITION_TOLERANCE_RAD,
    play_sounds: bool = True,
) -> dict[str, Any]:
    """Replay episodes and compare with recorded observations.
    
    Args:
        dataset: Recorded dataset
        robot_config: Robot configuration for replay
        episode_indices: List of episode indices to replay (None = all)
        tolerance: Maximum allowed joint error in radians
        play_sounds: Use vocal synthesis for events
        
    Returns:
        Dictionary with calibration test results
    """
    print(f"\n{'='*60}")
    print("STEP 2: Replaying Episodes and Testing Calibration")
    print(f"{'='*60}")
    
    # Initialize robot
    robot = make_robot_from_config(robot_config)
    robot_action_processor = make_default_processors()[1]  # robot_action_processor
    
    # Connect robot
    robot.connect()
    
    if not robot.is_connected:
        raise ValueError("Robot is not connected!")
    
    # Determine which episodes to replay
    if episode_indices is None:
        episode_indices = list(range(dataset.num_episodes))
    
    all_results = []
    
    for ep_idx in episode_indices:
        print(f"\nReplaying episode {ep_idx}...")
        log_say(f"Replaying episode {ep_idx}", play_sounds, blocking=False)
        
        # Filter dataset to episode
        episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == ep_idx)
        actions = episode_frames.select_columns(ACTION)
        observations = episode_frames.select_columns([f"{OBS_STR}.state"])
        
        episode_errors = []
        max_errors_per_frame = []
        
        for idx in range(len(episode_frames)):
            t0 = time.perf_counter()
            
            # Get recorded action
            action_array = actions[idx][ACTION]
            action = {
                name: float(action_array[i])
                for i, name in enumerate(dataset.features[ACTION]["names"])
            }
            
            # Get robot observation
            robot_obs = robot.get_observation()
            
            # Process action
            processed_action = robot_action_processor((action, robot_obs))
            
            # Send action to robot
            _ = robot.send_action(processed_action)
            
            # Get replayed observation - convert robot observation dict to array
            # Robot observation is a dict, we need to extract state values
            robot_state_values = []
            for key in sorted(robot_obs.keys()):
                val = robot_obs[key]
                if isinstance(val, (int, float)):
                    robot_state_values.append(float(val))
                elif isinstance(val, (list, tuple, np.ndarray)):
                    robot_state_values.extend([float(v) for v in val])
            
            replayed_obs = {"observation.state": np.array(robot_state_values, dtype=np.float32)}
            
            # Get recorded observation
            recorded_state = observations[idx][f"{OBS_STR}.state"]
            recorded_obs = {"observation.state": recorded_state}
            
            # Compare observations
            errors = compare_observations(recorded_obs, replayed_obs, tolerance)
            episode_errors.append(errors)
            
            if "state" in errors:
                max_error = errors["state"]["max_error"]
                max_errors_per_frame.append(max_error)
                
                # Print warning if error exceeds tolerance
                if max_error > tolerance:
                    print(f"  Frame {idx}: Max joint error {max_error:.4f} rad exceeds tolerance {tolerance:.4f}")
            
            # Maintain FPS
            precise_sleep(1.0 / dataset.fps - (time.perf_counter() - t0))
        
        # Episode summary
        if max_errors_per_frame:
            max_errors = np.array(max_errors_per_frame)
            episode_result = {
                "episode_index": ep_idx,
                "max_error": np.max(max_errors),
                "mean_max_error": np.mean(max_errors),
                "std_max_error": np.std(max_errors),
                "frames_exceeding_tolerance": np.sum(max_errors > tolerance),
                "total_frames": len(max_errors),
                "within_tolerance": np.max(max_errors) < tolerance,
            }
            all_results.append(episode_result)
            
            print(f"\n  Episode {ep_idx} Results:")
            print(f"    Max joint error: {episode_result['max_error']:.4f} rad ({np.degrees(episode_result['max_error']):.2f}°)")
            print(f"    Mean max error: {episode_result['mean_max_error']:.4f} rad")
            print(f"    Frames exceeding tolerance: {episode_result['frames_exceeding_tolerance']}/{episode_result['total_frames']}")
            print(f"    Status: {'✓ PASS' if episode_result['within_tolerance'] else '✗ FAIL'}")
    
    robot.disconnect()
    
    # Overall summary
    print(f"\n{'='*60}")
    print("CALIBRATION TEST SUMMARY")
    print(f"{'='*60}")
    
    if all_results:
        all_max_errors = [r["max_error"] for r in all_results]
        overall_max = np.max(all_max_errors)
        overall_mean = np.mean(all_max_errors)
        total_frames_exceeding = sum(r["frames_exceeding_tolerance"] for r in all_results)
        total_frames = sum(r["total_frames"] for r in all_results)
        all_pass = all(r["within_tolerance"] for r in all_results)
        
        print(f"\nOverall Results:")
        print(f"  Max joint error across all episodes: {overall_max:.4f} rad ({np.degrees(overall_max):.2f}°)")
        print(f"  Mean max error per episode: {overall_mean:.4f} rad ({np.degrees(overall_mean):.2f}°)")
        print(f"  Total frames exceeding tolerance: {total_frames_exceeding}/{total_frames}")
        print(f"  Tolerance threshold: {tolerance:.4f} rad ({np.degrees(tolerance):.2f}°)")
        
        print(f"\n{'='*60}")
        if all_pass and overall_max < MAX_JOINT_ERROR_TOLERANCE_RAD:
            print("✓ CALIBRATION TEST PASSED")
            print("  Your robot calibration looks good!")
        elif overall_max < MAX_JOINT_ERROR_TOLERANCE_RAD:
            print("⚠ CALIBRATION TEST PASSED WITH WARNINGS")
            print("  Some frames exceeded tolerance, but overall calibration is acceptable.")
        else:
            print("✗ CALIBRATION TEST FAILED")
            print("  Calibration may need adjustment. Errors exceed acceptable threshold.")
        print(f"{'='*60}\n")
        
        return {
            "overall_max_error": overall_max,
            "overall_mean_error": overall_mean,
            "total_frames_exceeding": total_frames_exceeding,
            "total_frames": total_frames,
            "passed": all_pass,
            "episode_results": all_results,
        }
    
    return {"passed": False, "episode_results": []}


@parser.wrap()
def main(cfg: CalibrationTestConfig):
    """Main calibration test function."""
    import logging
    
    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    # Set default dataset config if not provided
    if cfg.dataset is None:
        cfg.dataset = DatasetTestConfig()
    
    # Dataset paths
    dataset_root = Path(cfg.dataset.root) if cfg.dataset.root else Path("./local_datasets")
    
    # Record episodes (unless skipped)
    if not cfg.dataset.skip_record:
        dataset = record_test_episode(
            robot_config=cfg.robot,
            teleop_config=cfg.teleop,
            dataset_config=cfg.dataset,
        )
    else:
        # Load existing dataset
        print(f"\nLoading existing dataset from {dataset_root}/{cfg.dataset.repo_id}...")
        dataset = LeRobotDataset(cfg.dataset.repo_id, root=dataset_root)
        print(f"  Loaded {dataset.num_episodes} episode(s), {dataset.num_frames} frames")
    
    # Replay and test calibration
    results = replay_and_test_calibration(
        dataset=dataset,
        robot_config=cfg.robot,
        episode_indices=cfg.dataset.test_episodes,
        tolerance=cfg.tolerance,
        play_sounds=cfg.play_sounds,
    )
    
    # Exit with appropriate code
    exit_code = 0 if results.get("passed", False) else 1
    return exit_code


if __name__ == "__main__":
    # Register third-party plugins if available (optional)
    #register_third_party_plugins()
    exit(main())
