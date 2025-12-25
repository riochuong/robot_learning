# LeRobot Recording Process and Data Format Guide

## Overview

LeRobot uses a standardized dataset format (v3.0) for recording robot demonstrations. This guide explains how the recording process works and the structure of the data format.

## Recording Process Flow

### High-Level Flow

```
Robot → Observation Processing → Action Generation (Teleop/Policy) → Action Processing → Robot Execution → Dataset Storage
```

### Detailed Recording Loop

The recording process follows this flow (from `lerobot_record.py`):

1. **Initialize Dataset**
   - Create empty `LeRobotDataset` with feature schema
   - Set up image/video writers for camera data
   - Configure processors for observations and actions

2. **For Each Episode:**
   ```
   For each frame (at specified FPS):
     a. Get robot observation (raw_obs)
     b. Process observation through robot_observation_processor → processed_obs
     c. Get action from teleoperator OR policy:
        - Teleop: teleop.get_action() → teleop_action_processor → processed_teleop_action
        - Policy: predict_action() → processed_policy_action
     d. Process action through robot_action_processor → robot_action_to_send
     e. Send action to robot (robot.send_action())
     f. Build dataset frame:
        - observation_frame = build_dataset_frame(processed_obs)
        - action_frame = build_dataset_frame(action_values)
        - frame = {observation_frame, action_frame, "task": task_string}
     g. Add frame to dataset buffer: dataset.add_frame(frame)
     h. Sleep to maintain FPS
   
   After episode completes:
     - Save episode: dataset.save_episode()
     - Encode videos (if using video format)
     - Update metadata
   ```

3. **Finalize Dataset**
   - Call `dataset.finalize()` to close parquet writers
   - Push to Hugging Face Hub (optional)

### Key Components

#### Processors
- **robot_observation_processor**: Processes raw robot observations (default: IdentityProcessor)
- **teleop_action_processor**: Processes raw teleoperator actions
- **robot_action_processor**: Processes actions before sending to robot

#### Frame Building
Frames are built using `build_dataset_frame()` which:
- Takes raw values (observations/actions) and dataset features
- Converts to numpy arrays matching feature specifications
- Handles different data types (float32, images, videos)

## Data Format (LeRobotDataset v3.0)

### Core Principles

1. **File-based storage**: Multiple episodes per Parquet/MP4 file (not one file per episode)
2. **Relational metadata**: Episode boundaries resolved through metadata, not filenames
3. **Efficient storage**: Large files for better performance at scale

### Why Parquet Format?

Parquet is crucial for LeRobot datasets because it provides several key advantages:

#### 1. **Memory Efficiency via Memory Mapping**
- **Zero-copy access**: Parquet files can be memory-mapped, allowing data access without loading entire files into RAM
- **PyArrow integration**: The `datasets` library uses PyArrow's memory mapping, enabling efficient access to large datasets
- **Scalability**: Can handle datasets with millions of episodes without exhausting RAM
- Code reference: `load_nested_dataset()` uses `Dataset.from_parquet()` which leverages memory mapping

#### 2. **Columnar Storage Benefits**
- **Column-oriented layout**: Data is stored column-by-column rather than row-by-row
- **Selective reading**: Can read only specific columns (e.g., just `action` or just `observation.state`) without loading entire rows
- **Efficient for ML**: Perfect for accessing specific features during training without loading unnecessary data

#### 3. **Compression**
- **Built-in compression**: Uses Snappy compression by default (`compression="snappy"`)
- **Dictionary encoding**: Repeated values are encoded efficiently (`use_dictionary=True`)
- **Storage savings**: Significantly reduces file sizes compared to uncompressed formats
- Example: A dataset that would be 100GB uncompressed might be only 20-30GB in Parquet

#### 4. **Predicate Pushdown**
- **Filtering at read time**: Can filter data (e.g., specific episodes) before loading into memory
- **Efficient queries**: PyArrow can push filters down to the file level, reading only relevant row groups
- Code example: `pa_ds.field("episode_index").isin(episodes)` filters at the Parquet level

#### 5. **Schema Enforcement**
- **Type safety**: Parquet enforces data types (float32, int64, etc.) ensuring consistency
- **Shape validation**: Schema includes shape information, preventing dimension mismatches
- **Version compatibility**: Schema versioning helps maintain compatibility across codebase versions

#### 6. **Interoperability**
- **Standard format**: Parquet is widely supported across Python, R, Java, C++, and more
- **Hugging Face integration**: Native support in `datasets` library
- **Cloud-friendly**: Works seamlessly with cloud storage (S3, GCS, Azure)

#### 7. **Incremental Writing**
- **Append support**: Can append new episodes to existing Parquet files efficiently
- **Streaming writes**: Supports writing data incrementally during recording
- **File size management**: Automatically creates new files when size limits are reached

#### 8. **Metadata Rich**
- **File-level metadata**: Each Parquet file contains schema and statistics
- **Row group statistics**: Enables efficient filtering and querying
- **Fast metadata reads**: Can read file metadata without loading data

#### Real-World Impact

For a typical robot learning dataset:
- **Without Parquet**: Loading 1M frames might require 50GB+ RAM
- **With Parquet**: Same dataset uses <1GB RAM via memory mapping
- **Storage**: 70-80% reduction in disk space due to compression
- **Loading speed**: 10-100x faster initialization compared to JSON/CSV formats

### Directory Structure

```
dataset_root/
├── meta/
│   ├── info.json              # Schema, FPS, codebase version, path templates
│   ├── stats.json             # Feature statistics (mean/std/min/max)
│   ├── tasks.parquet          # Task descriptions mapped to IDs
│   └── episodes/
│       └── chunk-{chunk_idx:03d}/
│           └── file-{file_idx:03d}.parquet  # Episode metadata
├── data/
│   └── chunk-{chunk_idx:03d}/
│       └── file-{file_idx:03d}.parquet      # Frame data (many episodes)
└── videos/                    # Optional: MP4 videos per camera
    └── {video_key}/
        └── chunk-{chunk_idx:03d}/
            └── file-{file_idx:03d}.mp4       # Video data (many episodes)
```

### Frame Structure

A single frame is a dictionary containing:

#### Required Fields
- `observation.*`: Robot observations (state, images, etc.)
- `action`: Robot actions (joint positions, velocities, etc.)
- `task`: Natural language task description (string)
- `timestamp`: Time in episode (float32, seconds)
- `frame_index`: Frame index within episode (int64, starts at 0)
- `episode_index`: Episode index in dataset (int64)
- `index`: Global frame index across entire dataset (int64)
- `task_index`: Task ID mapped from task string (int64)

#### Example Frame Structure

```python
frame = {
    # Observations
    "observation.state": np.array([...], dtype=np.float32),  # Joint positions, etc.
    "observation.images.camera_name": np.ndarray,  # Image array or path
    # Actions
    "action": np.array([...], dtype=np.float32),  # Joint targets, velocities, etc.
    # Metadata
    "task": "Grab the cube",  # Natural language task description
    "timestamp": 1.234,  # Seconds into episode
    "frame_index": 42,  # Frame number in episode
    "episode_index": 5,  # Episode number
    "index": 1234,  # Global frame index
    "task_index": 0,  # Task ID
}
```

### Feature Types

Features are defined with:
- **dtype**: `float32`, `int64`, `bool`, `image`, `video`
- **shape**: Tuple describing dimensions
- **names**: Optional names for vector dimensions

#### Common Feature Patterns

```python
features = {
    # State vector
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),  # 7 joint positions
        "names": ["joint_0", "joint_1", ..., "joint_6"]
    },
    # Image from camera
    "observation.images.front_camera": {
        "dtype": "video",  # or "image"
        "shape": (3, 480, 640),  # C, H, W
        "names": None
    },
    # Action vector
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["target_joint_0", ..., "target_joint_6"]
    }
}
```

### Storage Details

#### Tabular Data (Parquet)
- Low-dimensional, high-frequency signals (states, actions, timestamps)
- Stored in Apache Parquet format
- Memory-mapped for efficient access
- Multiple episodes concatenated per file

#### Visual Data (MP4/Images)
- Camera frames encoded as MP4 videos (default) or PNG images
- Videos are sharded per camera
- Multiple episodes concatenated per video file
- Frame timestamps stored in parquet metadata

#### Metadata (JSON/Parquet)
- **info.json**: Schema, FPS, codebase version, path templates
- **stats.json**: Normalization statistics per feature
- **tasks.parquet**: Task descriptions and mappings
- **episodes/*.parquet**: Episode metadata (lengths, tasks, file offsets)

### Episode Metadata

Each episode has metadata stored in `meta/episodes/`:

```python
{
    "episode_index": 5,
    "length": 1800,  # Number of frames
    "tasks": ["Grab the cube"],  # Task descriptions
    "dataset_from_index": 9000,  # Start index in dataset
    "dataset_to_index": 10800,   # End index in dataset
    "data/chunk_index": 0,        # Parquet file location
    "data/file_index": 2,
    "videos/{camera}/chunk_index": 0,  # Video file location
    "videos/{camera}/file_index": 1,
}
```

## Recording API Usage

### Basic Recording Command

```bash
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --dataset.repo_id=<username>/<dataset_name> \
    --dataset.num_episodes=50 \
    --dataset.single_task="Grab the cube" \
    --dataset.fps=30 \
    --dataset.video=true
```

### Programmatic Recording

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config

# Create dataset
dataset = LeRobotDataset.create(
    repo_id="username/dataset_name",
    fps=30,
    robot_type="so100_follower",
    features=dataset_features,
    use_videos=True
)

# Initialize robot and teleop
robot = make_robot_from_config(robot_config)
teleop = make_teleoperator_from_config(teleop_config)

robot.connect()
teleop.connect()

# Recording loop
for episode in range(num_episodes):
    for frame in range(episode_length):
        # Get observation
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        
        # Get action
        action = teleop.get_action()
        action_processed = teleop_action_processor((action, obs))
        
        # Send to robot
        robot.send_action(action_processed)
        
        # Build and save frame
        observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix="observation")
        action_frame = build_dataset_frame(dataset.features, action_processed, prefix="action")
        frame = {**observation_frame, **action_frame, "task": "Grab the cube"}
        dataset.add_frame(frame)
    
    # Save episode
    dataset.save_episode()

# Finalize and push
dataset.finalize()
dataset.push_to_hub()
```

## Data Loading

### Loading a Dataset

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load from Hub or local path
dataset = LeRobotDataset("username/dataset_name")

# Access a frame
frame = dataset[100]  # Returns dict with tensors

# Access with temporal windows
delta_timestamps = {
    "observation.images.camera": [-0.2, -0.1, 0.0]  # 3 frames
}
dataset = LeRobotDataset("username/dataset_name", delta_timestamps=delta_timestamps)
frame = dataset[100]  # Returns stacked frames
```

### Streaming (No Download)

```python
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

# Stream directly from Hub
dataset = StreamingLeRobotDataset("username/dataset_name")
```

## Key Implementation Details

### Frame Buffer

During recording, frames are buffered in memory:
- `dataset.add_frame()` adds to `episode_buffer`
- Images are written to temporary directory immediately
- Other data stays in memory until `save_episode()`

### Video Encoding

- Images are first saved as PNG files
- After episode completes, images are encoded to MP4
- Videos can be encoded immediately or batched (configurable)
- Temporary PNG files are deleted after encoding

### Episode Saving

`save_episode()` does:
1. Validates episode buffer
2. Computes episode statistics
3. Saves frame data to parquet
4. Encodes videos (if using video format)
5. Updates episode metadata
6. Clears episode buffer

### Metadata Management

- Episode metadata is buffered and flushed periodically
- Parquet writers are managed per chunk/file
- Must call `finalize()` before pushing to Hub to close writers properly

## Best Practices

1. **Always call `finalize()`** before pushing to Hub
2. **Use video format** for better storage efficiency
3. **Set appropriate FPS** based on your control frequency
4. **Validate frames** match feature schema before adding
5. **Use processors** to transform data consistently
6. **Batch video encoding** for better performance with many episodes

## References

- Main recording script: `src/lerobot/scripts/lerobot_record.py`
- Dataset class: `src/lerobot/datasets/lerobot_dataset.py`
- Dataset utilities: `src/lerobot/datasets/utils.py`
- Documentation: `docs/source/lerobot-dataset-v3.mdx`

