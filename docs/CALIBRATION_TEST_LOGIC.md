# Complete Logic Explanation of `test_calibration.py`

## **Purpose**
Tests robot calibration accuracy by recording robot movements and then replaying them to see if the robot reaches the same positions. This validates that calibration is consistent and accurate.

---

## **🏗️ Script Structure Overview**

The script has 5 main components:
1. **Initialization** - Setup and imports
2. **Configuration Classes** - Define parameters
3. **Recording Function** - Capture robot movements
4. **Replay & Test Function** - Replay and compare
5. **Main Function** - Orchestrate everything

---

## **1. Initialization (Lines 1-79)**

### Purpose
Set up the environment and define constants.

```python
# Force offline mode - prevents trying to access HuggingFace hub
os.environ["HF_HUB_OFFLINE"] = "1"

# Import robot modules to register types
from lerobot.robots import so100_follower, so101_follower

# Constants
JOINT_POSITION_TOLERANCE_RAD = 0.05  # ~2.9 degrees acceptable error
MAX_JOINT_ERROR_TOLERANCE_RAD = 0.1  # ~5.7 degrees
DEFAULT_DATASET_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot"
```

### Why?
- **Offline mode**: Works without internet, uses local datasets only
- **Register robot types**: Makes robot types available to the command-line parser
- **Tolerance constants**: Define what error level is acceptable for calibration

---

## **2. Configuration Classes (Lines 81-168)**

### `compare_observations()` Function (Lines 81-135)

Compares recorded vs replayed joint positions.

```python
def compare_observations(recorded_obs, replayed_obs, tolerance, values_in_degrees=True):
    # 1. Extract state arrays from both observations
    recorded_state = recorded_obs["observation.state"]
    replayed_state = replayed_obs["observation.state"]
    
    # 2. Calculate absolute error (difference between positions)
    state_error = np.abs(recorded_state - replayed_state)
    
    # 3. Convert degrees to radians if needed (for proper tolerance comparison)
    if values_in_degrees:
        state_error_rad = np.deg2rad(state_error)
    
    # 4. Return error metrics
    return {
        "max_error": np.max(state_error_rad),      # Worst joint error (radians)
        "mean_error": np.mean(state_error_rad),    # Average error (radians)
        "within_tolerance": max < tolerance,       # Pass/fail boolean
        "max_error_deg": np.max(state_error),      # Error in degrees for display
        "errors": state_error_rad                  # Per-joint errors
    }
```

**Key Insight:** Values are stored in degrees but tolerance is in radians, so conversion is critical!

### Configuration Dataclasses (Lines 138-168)

#### `DatasetTestConfig`
Recording and replay settings:
- `repo_id`: Dataset name/path
- `num_episodes`: How many episodes to record
- `episode_time_s`: Duration of each episode
- `fps`: Recording framerate (30 Hz)
- `test_episodes`: Comma-separated episode indices to test (e.g., "0,1,2")
- `skip_record`: If True, skip recording and load existing dataset

#### `CalibrationTestConfig`
Overall test configuration:
- `robot`: Robot configuration (type, port, calibration)
- `teleop`: Teleoperator configuration (optional)
- `dataset`: Dataset configuration
- `tolerance`: Acceptable error threshold (radians)
- `play_sounds`: Enable audio feedback

---

## **3. Recording Function (Lines 171-310)**

### Purpose
Record robot movements controlled by teleoperation or a policy.

### Detailed Flow

```python
def record_test_episode(robot_config, teleop_config, dataset_config):
    # ============== STEP 1: Initialize Hardware ==============
    robot = make_robot_from_config(robot_config)        # Create robot interface
    teleop = make_teleoperator_from_config(teleop_config)  # Create leader arm
    
    # ============== STEP 2: Setup Data Processors ==============
    # These transform raw robot data into standardized format
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    
    # ============== STEP 3: Create Dataset Structure ==============
    # Define what features (joints) we're recording
    action_features = hw_to_dataset_features(robot.action_features, ACTION, use_video=False)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False)
    
    # Combine into dataset schema
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(teleop_action_processor, ...),
        aggregate_pipeline_dataset_features(robot_observation_processor, ...)
    )
    
    # Create the dataset
    dataset = LeRobotDataset.create(
        repo_id=dataset_config.repo_id,
        fps=dataset_config.fps,
        features=dataset_features,
        use_videos=False  # No cameras for calibration test
    )
    
    # ============== STEP 4: Connect Hardware ==============
    robot.connect()
    teleop.connect()
    
    # ============== STEP 5: Recording Loop ==============
    # Record multiple episodes
    for episode_idx in range(dataset_config.num_episodes):
        start_time = time.perf_counter()
        
        # Record frames at 30 FPS for specified duration
        while (time.perf_counter() - start_time) < dataset_config.episode_time_s:
            frame_start = time.perf_counter()
            
            # 5.1: Get current robot state
            obs = robot.get_observation()  # Raw joint positions
            obs_processed = robot_observation_processor(obs)  # Standardize format
            
            # 5.2: Get desired action from teleoperator
            action = teleop.get_action()  # Desired joint positions
            action_processed = teleop_action_processor((action, obs))  # Process
            
            # 5.3: Send action to robot
            robot_action = robot_action_processor((action_processed, obs))
            robot.send_action(robot_action)
            
            # 5.4: Save to dataset
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset.features, action_processed, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": dataset_config.task}
            dataset.add_frame(frame)
            
            # 5.5: Maintain precise timing (30 FPS)
            dt = time.perf_counter() - frame_start
            precise_sleep(1.0 / dataset_config.fps - dt)
        
        # Save episode to disk
        dataset.save_episode()
    
    # ============== STEP 6: Finalize ==============
    dataset.finalize()  # Write metadata
    robot.disconnect()
    teleop.disconnect()
    
    return dataset
```

### Key Points

1. **Observation → Action → Send → Save**: The control loop
2. **`build_dataset_frame()`**: Ensures data is in consistent format with proper transformations
3. **30 FPS timing**: Uses `precise_sleep()` to maintain exact framerate
4. **No cameras**: Only joint positions are recorded for calibration testing

---

## **4. Replay & Test Function (Lines 313-543)**

### Purpose
Replay recorded actions and compare if robot reaches the same positions.

### Detailed Flow

```python
def replay_and_test_calibration(dataset, robot_config, episode_indices, tolerance):
    # ============== STEP 1: Initialize ==============
    robot = make_robot_from_config(robot_config)
    _, robot_action_processor, robot_observation_processor = make_default_processors()
    robot.connect()
    
    # Determine which episodes to test
    if episode_indices is None:
        episode_indices = list(range(dataset.num_episodes))
    
    all_results = []
    
    # ============== STEP 2: Loop Through Episodes ==============
    for ep_idx in episode_indices:
        # Load episode data
        episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == ep_idx)
        actions = episode_frames.select_columns(ACTION)
        observations = episode_frames.select_columns("observation.state")
        
        # ============== STEP 3: CRITICAL - Move to Starting Position ==============
        # Robot must start where recording started!
        initial_recorded_state = observations[0]["observation.state"]
        
        # Create action to move to start
        initial_action = {}
        for i, name in enumerate(dataset.features[ACTION]["names"]):
            initial_action[name] = float(initial_recorded_state[i])
        
        # Send and wait for robot to reach position
        robot_obs_init = robot.get_observation()
        processed_initial_action = robot_action_processor((initial_action, robot_obs_init))
        robot.send_action(processed_initial_action)
        time.sleep(1.0)  # Wait for robot to settle
        
        episode_errors = []
        max_errors_per_frame = []
        
        # ============== STEP 4: Replay Loop ==============
        for idx in range(len(episode_frames)):
            t0 = time.perf_counter()
            
            # 4.1: Load recorded action
            action_array = actions[idx][ACTION]
            action = {
                name: float(action_array[i])
                for i, name in enumerate(dataset.features[ACTION]["names"])
            }
            
            # 4.2: Get observation BEFORE (for processing)
            robot_obs_before = robot.get_observation()
            
            # 4.3: Process and send action
            processed_action = robot_action_processor((action, robot_obs_before))
            robot.send_action(processed_action)
            
            # 4.4: Get observation AFTER action is applied
            robot_obs = robot.get_observation()
            
            # 4.5: Process observation through same pipeline as recording
            obs_processed = robot_observation_processor(robot_obs)
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
            replayed_state = observation_frame["observation.state"]
            
            # 4.6: Load what was recorded
            recorded_state = observations[idx]["observation.state"]
            
            # 4.7: Compare!
            errors = compare_observations(
                {"observation.state": recorded_state},
                {"observation.state": replayed_state},
                tolerance,
                values_in_degrees=True  # Values are in degrees
            )
            
            episode_errors.append(errors)
            max_errors_per_frame.append(errors["state"]["max_error"])
            
            # 4.8: Report if error exceeds tolerance
            if errors["state"]["max_error"] > tolerance:
                print(f"Frame {idx}: Error {errors['state']['max_error_deg']:.2f}° exceeds tolerance")
            
            # 4.9: Maintain timing
            precise_sleep(1.0 / dataset.fps - (time.perf_counter() - t0))
        
        # ============== STEP 5: Calculate Episode Statistics ==============
        max_errors = np.array(max_errors_per_frame)
        episode_result = {
            "episode_index": ep_idx,
            "max_error": np.max(max_errors),
            "mean_max_error": np.mean(max_errors),
            "frames_exceeding_tolerance": np.sum(max_errors > tolerance),
            "total_frames": len(max_errors),
            "within_tolerance": np.max(max_errors) < tolerance
        }
        all_results.append(episode_result)
    
    robot.disconnect()
    
    # ============== STEP 6: Overall Summary ==============
    all_max_errors = [r["max_error"] for r in all_results]
    overall_max = np.max(all_max_errors)
    all_pass = all(r["within_tolerance"] for r in all_results)
    
    # Determine pass/fail
    if all_pass and overall_max < MAX_JOINT_ERROR_TOLERANCE_RAD:
        print("✓ CALIBRATION TEST PASSED")
    else:
        print("✗ CALIBRATION TEST FAILED")
    
    return {
        "overall_max_error": overall_max,
        "total_frames_exceeding": sum(r["frames_exceeding_tolerance"] for r in all_results),
        "passed": all_pass,
        "episode_results": all_results
    }
```

### Critical Steps Explained

#### Why Move to Starting Position? (Step 3)

```
Problem:
  Recording Frame 0: Robot at position A (e.g., [10°, -90°, 95°, ...])
  Replay Frame 0: Robot at position B (e.g., [-5°, 65°, -20°, ...])
  ❌ Comparing these directly = huge error (150°+)

Solution:
  Before replay, move robot to position A
  ✅ Now both start from same position, fair comparison
```

#### Why Get Observation AFTER Action? (Step 4.4)

```python
# WRONG ORDER:
obs = robot.get_observation()  # Position BEFORE action
robot.send_action(action)
# Compare obs with recorded → MISMATCH (1 frame offset)

# CORRECT ORDER:
obs_before = robot.get_observation()  # For processing
robot.send_action(action)
obs_after = robot.get_observation()  # Position AFTER action
# Compare obs_after with recorded → MATCH
```

#### Why Use `build_dataset_frame()`? (Step 4.5)

```python
# Option 1: Direct comparison
raw_obs = robot.get_observation()  # {'shoulder_pan.pos': 11.46, ...}
# ❌ Format might not match dataset

# Option 2: Process like during recording
raw_obs = robot.get_observation()
processed = robot_observation_processor(raw_obs)
dataset_frame = build_dataset_frame(dataset.features, processed)
# ✅ Same transformations as recording = fair comparison
```

`build_dataset_frame()` applies:
- Normalization
- Unit conversions
- Feature ordering
- Format standardization

---

## **5. Main Function (Lines 546-649)**

### Purpose
Orchestrate the entire test process based on command-line arguments.

### Flow

```python
@parser.wrap()
def main(cfg: CalibrationTestConfig):
    # ============== STEP 1: Environment Setup ==============
    os.environ["HF_HUB_OFFLINE"] = "1"  # Force offline mode
    init_logging()
    
    # ============== STEP 2: Auto-detect Calibration File ==============
    if cfg.robot.id and not cfg.robot.calibration_dir:
        # Look for: ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/{robot.id}.json
        robot_type_name = type(cfg.robot).__name__.lower()
        
        for pattern in [robot_type_name, "so101_follower"]:
            calibration_dir = DEFAULT_DATASET_ROOT / "calibration" / "robots" / pattern
            calibration_file = calibration_dir / f"{cfg.robot.id}.json"
            
            if calibration_file.exists():
                cfg.robot.calibration_dir = calibration_dir
                print(f"✓ Found calibration file: {calibration_file}")
                break
    
    # ============== STEP 3: Setup Dataset Paths ==============
    if cfg.dataset.root:
        dataset_root = Path(cfg.dataset.root)
    else:
        dataset_root = DEFAULT_DATASET_ROOT  # ~/.cache/huggingface/lerobot
    
    # ============== STEP 4: Parse Episode Indices ==============
    # Convert "0,1,2" → [0, 1, 2]
    episode_indices = None
    if cfg.dataset.test_episodes:
        episode_indices = [int(x.strip()) for x in cfg.dataset.test_episodes.split(",") if x.strip()]
    
    # ============== STEP 5: Branch Based on Mode ==============
    if not cfg.dataset.skip_record:
        # RECORDING MODE: Create new dataset
        print("Recording new episodes...")
        dataset = record_test_episode(
            robot_config=cfg.robot,
            teleop_config=cfg.teleop,
            dataset_config=cfg.dataset
        )
    else:
        # REPLAY MODE: Load existing dataset
        dataset_path = dataset_root / cfg.dataset.repo_id
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        print(f"Loading existing dataset from {dataset_path}...")
        dataset = LeRobotDataset(cfg.dataset.repo_id, root=str(dataset_path.absolute()))
    
    # ============== STEP 6: Test Calibration ==============
    results = replay_and_test_calibration(
        dataset=dataset,
        robot_config=cfg.robot,
        episode_indices=episode_indices,
        tolerance=cfg.tolerance,
        play_sounds=cfg.play_sounds
    )
    
    # ============== STEP 7: Exit with Status Code ==============
    # Exit code 0 = success (test passed)
    # Exit code 1 = failure (test failed)
    exit_code = 0 if results.get("passed", False) else 1
    return exit_code
```

### Command-Line Usage

```bash
# Recording mode (with teleoperation)
uv run python test_calibration.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower_arm_so_101 \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --dataset.repo_id=calibration_test \
    --dataset.num_episodes=3 \
    --dataset.episode_time_s=10

# Replay mode (test existing dataset)
uv run python test_calibration.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower_arm_so_101 \
    --dataset.repo_id=calibration_test \
    --dataset.skip_record=True \
    --dataset.test_episodes=0,1,2
```

---

## **🔑 Key Concepts Explained**

### 1. Why Starting Position Matters

**Problem:**
- During recording: Robot starts at position A
- During replay: Robot starts at position B (different!)
- Comparing them directly: Massive error even with perfect calibration

**Solution:**
```python
# Before replay loop:
initial_position = observations[0]  # Where robot was at start of recording
robot.send_action(initial_position)  # Move to that position
time.sleep(1.0)  # Wait to settle
# Now replay starts from same position as recording
```

**Example:**
```
Without initial positioning:
  Recorded[0]: [10°, -90°, 95°, ...]  
  Replayed[0]: [-5°, 65°, -20°, ...]  
  Error: 150°+ ❌

With initial positioning:
  Move robot to [10°, -90°, 95°, ...]
  Recorded[0]: [10°, -90°, 95°, ...]
  Replayed[0]: [10.2°, -89.8°, 94.9°, ...]
  Error: 0.5° ✓
```

### 2. Observation Processing Pipeline

Raw robot data must go through same transformations as during recording:

```
robot.get_observation()
  ↓ (raw joint angles)
robot_observation_processor()
  ↓ (standardized format)
build_dataset_frame()
  ↓ (normalized, ordered, transformed)
Final dataset value
```

**Why?** If recording used transformations but replay doesn't, values won't match even if robot is in correct position.

### 3. Unit Conversion

**The Problem:**
- Dataset stores values in **degrees**: `[11.46, -98.88, 99.16, ...]`
- Tolerance defined in **radians**: `0.05 rad` (~2.9°)

**The Solution:**
```python
# Calculate error in degrees
error_deg = abs(recorded - replayed)  # e.g., 0.66°

# Convert to radians for comparison
error_rad = np.deg2rad(error_deg)  # 0.66° = 0.0115 rad

# Compare with tolerance
if error_rad < tolerance:  # 0.0115 < 0.05
    print("✓ PASS")  # Calibration is good!
```

### 4. Timing and Frame Synchronization

```python
# Maintain precise 30 FPS:
for frame in frames:
    frame_start = time.perf_counter()
    
    # Do work (read obs, send action, etc.)
    process_frame()
    
    # Calculate how long work took
    elapsed = time.perf_counter() - frame_start
    
    # Sleep for remaining time to hit exactly 1/30 second
    precise_sleep(1.0 / 30.0 - elapsed)
```

This ensures:
- Consistent timing between recording and replay
- Fair comparison (same speeds)
- Robot has time to reach target position

---

## **📊 Data Flow Diagrams**

### Recording Flow

```
┌─────────────────┐
│  Teleoperator   │ ← Human controls leader arm
└────────┬────────┘
         │ desired action
         ↓
┌─────────────────┐
│ Action Processor│ ← Standardize format
└────────┬────────┘
         │ processed action
         ↓
┌─────────────────┐
│  robot.send()   │ ← Send to follower arm
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  robot.get()    │ ← Read current position
└────────┬────────┘
         │ observation
         ↓
┌─────────────────┐
│ Obs Processor   │ ← Standardize format
└────────┬────────┘
         │ processed obs
         ↓
┌─────────────────┐
│build_dataset_   │ ← Apply transformations
│    frame()      │
└────────┬────────┘
         │ final values
         ↓
┌─────────────────┐
│  dataset.add()  │ ← Save to disk
└─────────────────┘
```

### Replay Flow

```
┌─────────────────┐
│ dataset.load()  │ ← Load recorded action
└────────┬────────┘
         │ action
         ↓
┌─────────────────┐
│ Action Processor│ ← Process
└────────┬────────┘
         │ processed action
         ↓
┌─────────────────┐
│  robot.send()   │ ← Send to robot
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  robot.get()    │ ← Read CURRENT position
└────────┬────────┘
         │ observation
         ↓
┌─────────────────┐
│ Obs Processor   │ ← Process
└────────┬────────┘
         │ processed obs
         ↓
┌─────────────────┐
│build_dataset_   │ ← Same transformations as recording
│    frame()      │
└────────┬────────┘
         │ current_values
         ↓
┌─────────────────┐
│ dataset.load()  │ ← Load recorded observation
└────────┬────────┘
         │ recorded_values
         ↓
┌─────────────────────────────┐
│ compare_observations()      │ ← Calculate error
│ error = |current - recorded|│
└────────────┬────────────────┘
             │
             ↓
      ┌──────────────┐
      │ error < tol? │
      └──────┬───────┘
             │
    ┌────────┴────────┐
    │                 │
   YES               NO
    │                 │
 ✓ PASS           ✗ FAIL
```

---

## **🎯 Summary**

### What the Script Does

The `test_calibration.py` script validates robot calibration by:

1. **Recording** robot movements with teleoperation
2. **Replaying** those exact movements
3. **Comparing** if robot reaches the same positions (within tolerance)

### Pass/Fail Criteria

- **✓ Excellent**: Errors < 0.05 rad (~2.9°)
- **⚠ Acceptable**: Errors 0.05-0.1 rad (~2.9-5.7°)  
- **✗ Failed**: Errors > 0.1 rad (~5.7°) → recalibration needed

### Most Important Parts

1. **Move to starting position** (Lines 372-389)
   - Robot must start where recording started
   - Otherwise, comparison is meaningless

2. **Use same processing pipeline** (Lines 416-420)
   - `build_dataset_frame()` must be used in both recording and replay
   - Ensures apples-to-apples comparison

3. **Unit conversion** (Lines 118-132)
   - Values in degrees, tolerance in radians
   - Must convert before comparing

4. **Timing** (Lines 404-414)
   - Get observation BEFORE action (for processing)
   - Get observation AFTER action (for comparison)
   - This is critical for accuracy

### Common Pitfalls & Solutions

| Issue | Symptom | Fix |
|-------|---------|-----|
| Different starting positions | Huge errors (100°+) | Move to initial position first |
| Wrong unit comparison | Errors reported as radians when they're degrees | Convert degrees → radians |
| Observation timing | 1-frame offset errors | Get obs AFTER sending action |
| Missing processing | Format mismatch | Use `build_dataset_frame()` |
| Calibration not loaded | Using default values | Check `--robot.id` is specified |

---

## **📝 Quick Reference**

### File Locations

```
~/.cache/huggingface/lerobot/
├── calibration/
│   └── robots/
│       └── so101_follower/
│           └── my_follower_arm_so_101.json  ← Calibration file
└── data/
    └── calibration_test/                    ← Dataset
        ├── meta/
        │   ├── info.json                    ← Dataset metadata
        │   └── stats.json                   ← Value ranges/stats
        └── data/
            └── chunk-000/
                └── file-000.parquet         ← Actual data
```

### Key Constants

```python
JOINT_POSITION_TOLERANCE_RAD = 0.05  # 2.9° - acceptable error per frame
MAX_JOINT_ERROR_TOLERANCE_RAD = 0.1  # 5.7° - max acceptable overall
FPS = 30                              # Recording/replay framerate
DEFAULT_DATASET_ROOT = "~/.cache/huggingface/lerobot"
```

### Error Interpretation

| Max Error | Status | Action |
|-----------|--------|--------|
| < 0.05 rad (2.9°) | ✓ PASS | Calibration is excellent |
| 0.05-0.1 rad (2.9-5.7°) | ⚠ WARNING | Calibration acceptable but could be better |
| > 0.1 rad (5.7°) | ✗ FAIL | Recalibration recommended |

---

## **🐛 Debugging Tips**

### If You Get Large Errors (>10°)

1. **Check starting position**: Enable debug output (first 3 frames print automatically)
   - Recorded and replayed should be very close at frame 0
   - If frame 0 has large error, starting position alignment failed

2. **Check units**: Look at raw values in debug output
   - Values should be in degrees (typically -180 to 180 range)
   - If values are very large (>1000), unit mismatch

3. **Check calibration is loaded**: Look for this message:
   ```
   ✓ Found calibration file: /path/to/{robot.id}.json
   ```
   - If not found, calibration may not be applied
   - Specify `--robot.id` to auto-load calibration

4. **Verify dataset format**: Check if dataset was recorded with same robot type
   - Dataset `info.json` should show correct `robot_type`

### Debug Output

The script prints detailed info for first 3 frames of each episode:
```
DEBUG - Frame 0:
  Raw robot obs: {'shoulder_pan.pos': 11.71, ...}
  Processed obs keys: [...]
  Dataset frame keys: [...]
  Replayed state: [11.71, -98.22, ...]
  Recorded state: [11.46, -98.88, ...]
  Lengths: recorded=6, replayed=6
```

Look for:
- **Shape mismatch**: Different number of joints
- **Value range mismatch**: One in degrees, other in radians
- **Large differences at frame 0**: Starting position not aligned

---

## **🔧 Troubleshooting Common Issues**

### Issue: "Repository Not Found" Error

**Cause:** Script trying to access HuggingFace hub

**Solution:** Already fixed - `HF_HUB_OFFLINE=1` is set at import time

### Issue: "invalid choice: 'so_101'"

**Cause:** Robot type not registered

**Solution:** Already fixed - robot modules imported at top:
```python
from lerobot.robots import so100_follower, so101_follower
```

### Issue: "invalid list[int] value: '0,1,2'"

**Cause:** Wrong argument type

**Solution:** Changed to string type, parsed manually:
```python
test_episodes: str = ""  # Comma-separated: "0,1,2"
# Later parsed: [int(x) for x in test_episodes.split(",")]
```

### Issue: Calibration Not Loading

**Cause:** Auto-detection fails

**Solution:** Either:
1. Ensure `--robot.id` matches calibration filename
2. Manually specify: `--robot.calibration_dir=/path/to/calibration/robots/so101_follower`

---

## **💡 Best Practices**

### For Accurate Calibration Testing

1. **Use consistent robot.id**
   - Same ID for recording and replay
   - Ensures same calibration is used

2. **Test multiple episodes**
   - At least 3 episodes
   - Different movement patterns
   - Helps identify systematic vs random errors

3. **Appropriate episode length**
   - 10-30 seconds per episode
   - Covers full range of motion
   - Not too long (slow to test)

4. **Check calibration file exists**
   - Before testing, verify: `~/.cache/huggingface/lerobot/calibration/robots/{robot_type}/{robot_id}.json`
   - Should contain homing offsets and ranges for each joint

5. **Understand tolerance**
   - 0.05 rad = 2.9° is quite strict
   - Adjust based on your robot's repeatability
   - Consider increasing if robot mechanically imprecise

---

## **📖 Understanding the Output**

### Example Good Output

```
Episode 0 Results:
  Max joint error: 0.0234 rad (1.34°)
  Mean max error: 0.0156 rad
  Frames exceeding tolerance: 0/300
  Status: ✓ PASS

Overall Results:
  Max joint error across all episodes: 0.0298 rad (1.71°)
  Mean max error per episode: 0.0245 rad
  Total frames exceeding tolerance: 0/900
  
✓ CALIBRATION TEST PASSED
  Your robot calibration looks good!
```

### Example Bad Output

```
Episode 0 Results:
  Max joint error: 0.1456 rad (8.34°)
  Mean max error: 0.0823 rad
  Frames exceeding tolerance: 234/300
  Status: ✗ FAIL

Overall Results:
  Max joint error across all episodes: 0.1567 rad (8.98°)
  Mean max error per episode: 0.0934 rad
  Total frames exceeding tolerance: 702/900

✗ CALIBRATION TEST FAILED
  Calibration may need adjustment. Errors exceed acceptable threshold.
```

**Actions:**
- Recalibrate the robot
- Check for mechanical issues (loose joints, wear)
- Verify correct calibration file is loaded

---

**Last Updated:** December 13, 2025  
**Script Version:** test_calibration.py (with offline mode and auto-calibration detection)
**Author:** Robot Learning Project

