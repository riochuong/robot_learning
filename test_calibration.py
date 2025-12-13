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

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

# Force offline mode to prevent HF hub access - use local datasets only
# Set this before importing lerobot modules to ensure it's respected
os.environ["HF_HUB_OFFLINE"] = "1"

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
JOINT_POSITION_TOLERANCE_DEG = 3.0  # degrees (alternative when working in degrees)

# Default dataset root directory for offline mode (HF cache location)
DEFAULT_DATASET_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot"


def compare_observations(
    recorded_obs: dict[str, Any],
    replayed_obs: dict[str, Any],
    tolerance: float = JOINT_POSITION_TOLERANCE_RAD,
    values_in_degrees: bool = True,
) -> dict[str, Any]:
    """Compare recorded and replayed observations.
    
    Args:
        recorded_obs: Observation from recorded dataset
        replayed_obs: Observation from robot during replay
        tolerance: Maximum allowed error in radians
        values_in_degrees: If True, convert degree errors to radians before comparison
        
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
        
        # Calculate error (in same units as input)
        state_error = np.abs(recorded_state - replayed_state)
        
        # Convert to radians if values are in degrees
        if values_in_degrees:
            state_error_rad = np.deg2rad(state_error)
        else:
            state_error_rad = state_error
        
        errors["state"] = {
            "max_error": np.max(state_error_rad),
            "mean_error": np.mean(state_error_rad),
            "errors": state_error_rad,
            "within_tolerance": np.max(state_error_rad) < tolerance,
            "max_error_deg": np.max(state_error) if values_in_degrees else np.degrees(np.max(state_error_rad)),
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
    # Episode indices to test as comma-separated string (e.g., "0,1,2"), empty string means all episodes
    test_episodes: str = ""
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
    
    # Create dataset locally - use HF cache directory as default if root not specified
    dataset_root = Path(dataset_config.root) if dataset_config.root else DEFAULT_DATASET_ROOT
    
    dataset = LeRobotDataset.create(
        repo_id=dataset_config.repo_id,
        fps=dataset_config.fps,
        root=str(dataset_root),
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
            
            # Debug: Print first frame details
            if episode_idx == 0 and frame_count == 0:
                print(f"\n  DEBUG - First recorded frame:")
                print(f"    obs_processed keys: {list(obs_processed.keys())}")
                print(f"    observation_frame keys: {list(observation_frame.keys())}")
                if f"{OBS_STR}.state" in observation_frame:
                    print(f"    observation.state value: {observation_frame[f'{OBS_STR}.state']}")
                    print(f"    observation.state shape/length: {len(observation_frame[f'{OBS_STR}.state'])}")
            
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
    # Get processors - these should match what was used during recording
    _, robot_action_processor, robot_observation_processor = make_default_processors()
    
    # Create a pipeline to process observations the same way they were recorded
    # This is critical for proper comparison
    print(f"\n  Dataset info:")
   # print(f"    Robot type: {dataset.robot_type}")
    print(f"    FPS: {dataset.fps}")
    print(f"    Features: {list(dataset.features.keys())}")
    
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
        
        print(f"  Total frames in episode: {len(episode_frames)}")
        print(f"  Observation state features: {dataset.features[f'{OBS_STR}.state']['names']}")
        
        # Get the initial position from the recorded data
        initial_recorded_state = observations[0][f"{OBS_STR}.state"]
        print(f"\n  Initial recorded position: {initial_recorded_state}")
        
        # Move robot to initial position before starting replay
        print(f"  Moving robot to initial position...")
        initial_action = {}
        for i, name in enumerate(dataset.features[ACTION]["names"]):
            initial_action[name] = float(initial_recorded_state[i])
        
        # Process and send initial action to move to start position
        robot_obs_init = robot.get_observation()
        processed_initial_action = robot_action_processor((initial_action, robot_obs_init))
        robot.send_action(processed_initial_action)
        
        # Wait a moment for robot to reach position
        time.sleep(1.0)
        print(f"  Robot moved to starting position. Beginning replay...\n")
        
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
            
            # Get robot observation BEFORE sending action (for processing)
            robot_obs_before = robot.get_observation()
            
            # Process action
            processed_action = robot_action_processor((action, robot_obs_before))
            
            # Send action to robot
            _ = robot.send_action(processed_action)
            
            # Get robot observation AFTER sending action
            robot_obs = robot.get_observation()
            
            # Process observation through the same pipeline used during recording
            obs_processed = robot_observation_processor(robot_obs)
            
            # Build dataset frame to match the recording format
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
            
            # Extract the state
            replayed_state = observation_frame.get(f"{OBS_STR}.state")
            
            if replayed_state is None:
                # Fallback: extract directly from obs_processed
                obs_state_names = dataset.features[f"{OBS_STR}.state"]["names"]
                robot_state_values = []
                for name in obs_state_names:
                    if name in obs_processed:
                        val = obs_processed[name]
                        if isinstance(val, (list, tuple, np.ndarray)):
                            robot_state_values.extend([float(v) for v in val])
                        else:
                            robot_state_values.append(float(val))
                replayed_state = np.array(robot_state_values, dtype=np.float32)
            
            replayed_obs = {"observation.state": replayed_state}
            
            # Debug output on first few frames
            if idx == 0:
                print(f"\n  DEBUG - Frame {idx}:")
                print(f"    Raw robot obs: {robot_obs}")
                print(f"    Processed obs keys: {list(obs_processed.keys())}")
                print(f"    Observation frame keys: {list(observation_frame.keys())}")
                print(f"    Replayed state: {replayed_state}")
            
            # Get recorded observation
            recorded_state = observations[idx][f"{OBS_STR}.state"]
            recorded_obs = {"observation.state": recorded_state}
            
            if idx == 0:
                print(f"    Recorded state: {recorded_state}")
                print(f"    Replayed state: {replayed_obs['observation.state']}")
                print(f"    Lengths: recorded={len(recorded_state)}, replayed={len(replayed_obs['observation.state'])}")
            
            # Compare observations (values are in degrees based on dataset stats)
            errors = compare_observations(recorded_obs, replayed_obs, tolerance, values_in_degrees=True)
            episode_errors.append(errors)
            
            if "state" in errors:
                max_error = errors["state"]["max_error"]
                max_errors_per_frame.append(max_error)
                
                # Print first few frames for debugging
                if idx < 3:
                    print(f"\n  Frame {idx} comparison:")
                    print(f"    Recorded state shape: {recorded_obs['observation.state'].shape}")
                    print(f"    Replayed state shape: {replayed_obs['observation.state'].shape}")
                    print(f"    Recorded: {recorded_obs['observation.state'][:5]}...")
                    print(f"    Replayed: {replayed_obs['observation.state'][:5]}...")
                    print(f"    Max error: {max_error:.4f} rad ({errors['state']['max_error_deg']:.2f}°)")
                
                # Print warning if error exceeds tolerance
                if max_error > tolerance:
                    print(f"  Frame {idx}: Max joint error {max_error:.4f} rad ({errors['state']['max_error_deg']:.2f}°) exceeds tolerance {tolerance:.4f} rad ({np.degrees(tolerance):.2f}°)")
            
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
    import os
    
    # Force offline mode to prevent HF hub access - use local datasets only
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    init_logging()
    
    # Set default calibration directory if robot.id is provided but calibration_dir is not
    # Calibration files are stored at: ~/.cache/huggingface/lerobot/calibration/robots/{robot_type}/{robot_id}.json
    if cfg.robot.id and not cfg.robot.calibration_dir:
        # Get robot type name from the config class
        try:
            # Use the type attribute directly since it's set by the parser
            robot_type_name = type(cfg.robot).__name__.replace('Config', '').replace('SO', 'so').lower()
            # Try common naming patterns
            for pattern in [robot_type_name, f"so{robot_type_name.replace('so', '')}", "so101_follower"]:
                calibration_dir = DEFAULT_DATASET_ROOT / "calibration" / "robots" / pattern
                calibration_file = calibration_dir / f"{cfg.robot.id}.json"
                if calibration_file.exists():
                    cfg.robot.calibration_dir = calibration_dir
                    print(f"\n✓ Found calibration file: {calibration_file}")
                    print(f"  Using calibration for robot.id={cfg.robot.id}")
                    break
            else:
                print(f"\n⚠ Calibration file not found for robot.id={cfg.robot.id}")
                print(f"  Searched in: {DEFAULT_DATASET_ROOT / 'calibration' / 'robots'}")
        except Exception as e:
            # If we can't determine robot type, skip auto-configuration
            print(f"\n⚠ Could not auto-detect calibration directory: {e}")
            print(f"  Please specify --robot.calibration_dir manually if needed")
    
    logging.info(pformat(asdict(cfg)))
    
    # Set default dataset config if not provided
    if cfg.dataset is None:
        cfg.dataset = DatasetTestConfig()
    
    # Dataset paths - use HF cache directory as default prefix for offline mode
    if cfg.dataset.root:
        dataset_root = Path(cfg.dataset.root)
    else:
        # Default to HF cache directory for offline/local datasets
        dataset_root = DEFAULT_DATASET_ROOT
    
    # Parse test_episodes from comma-separated string to list of integers
    episode_indices = None
    if cfg.dataset.test_episodes:
        episode_indices = [int(x.strip()) for x in cfg.dataset.test_episodes.split(",") if x.strip()]
    
    # Record episodes (unless skipped)
    if not cfg.dataset.skip_record:
        dataset = record_test_episode(
            robot_config=cfg.robot,
            teleop_config=cfg.teleop,
            dataset_config=cfg.dataset,
        )
    else:
        # Load existing dataset locally (offline mode already set above)
        dataset_path = dataset_root / cfg.dataset.repo_id
        print(f"\nLoading existing dataset from {dataset_path}...")
        
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. "
                f"Make sure the dataset exists locally or record it first by removing --dataset.skip_record"
            )
        
        # Load from local path only (HF_HUB_OFFLINE is set at top of function to prevent hub access)
        try:
            dataset = LeRobotDataset(cfg.dataset.repo_id, root=str(dataset_path.absolute()))
        except Exception as e:
            if "huggingface" in str(e).lower() or "repository not found" in str(e).lower():
                raise RuntimeError(
                    f"Failed to load dataset: {e}\n"
                    f"Make sure the dataset exists locally at {dataset_path}.\n"
                    f"The script is configured to work offline only. "
                    f"If you need to download from HuggingFace, do so manually first."
                ) from e
            raise
        print(f"  Loaded {dataset.num_episodes} episode(s), {dataset.num_frames} frames")
    
    # Replay and test calibration
    results = replay_and_test_calibration(
        dataset=dataset,
        robot_config=cfg.robot,
        episode_indices=episode_indices,
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
