# LeRobot Data Collection Pipeline — Deep Dive

A comprehensive guide to understanding how robot demonstrations are captured, timestamped, and stored in the LeRobot framework — from bits on the wire to training-ready datasets.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Layer 1: Hardware Communication](#layer-1-hardware-communication-bits-on-the-wire)
  - [Motor Bus Communication](#11-motor-bus-communication)
  - [Camera Capture](#12-camera-capture)
- [Layer 2: Robot Abstraction](#layer-2-robot-abstraction)
- [Layer 3: Recording Loop](#layer-3-recording-loop)
- [Layer 4: In-Memory Buffering](#layer-4-in-memory-buffering-add_frame)
- [Layer 5: Episode Finalization](#layer-5-episode-finalization-save_episode)
- [Layer 6: Dataset Structure on Disk](#layer-6-final-dataset-structure-on-disk)
- [Layer 7: Video Timestamp Resolution](#layer-7-video-timestamp-computation-during-training)
- [Complete Data Flow Summary](#summary-complete-data-flow)
- [Common Issues & Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RECORDING LOOP (lerobot_record.py)                │
│                              @ configured FPS (e.g., 30 Hz)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
┌───────────────┐           ┌───────────────────┐           ┌───────────────┐
│   CAMERAS     │           │     ROBOT         │           │  TELEOPERATOR │
│  (OpenCV,     │           │  (SO100, LeKiwi,  │           │  (Leader arm, │
│   RealSense)  │           │   Unitree G1...)  │           │   Keyboard)   │
└───────┬───────┘           └────────┬──────────┘           └───────┬───────┘
        │ USB/Network                │ Serial/USB/DDS               │ Serial/USB
        ▼                            ▼                              ▼
┌───────────────┐           ┌───────────────────┐           ┌───────────────┐
│ camera.read() │           │ bus.sync_read()   │           │ teleop.       │
│ → numpy array │           │ → motor positions │           │ get_action()  │
└───────┬───────┘           └────────┬──────────┘           └───────┬───────┘
        └─────────────────┬──────────┴──────────────────────────────┘
                          ▼
                ┌─────────────────────┐
                │ robot.get_         │──► obs_dict
                │ observation()      │
                └─────────┬──────────┘
                          ▼
                ┌─────────────────────┐
                │ build_dataset_     │──► frame dict
                │ frame()            │
                └─────────┬──────────┘
                          ▼
                ┌─────────────────────┐
                │ dataset.add_frame()│──► episode_buffer + temp PNGs
                └─────────┬──────────┘
                          │ (end of episode)
                          ▼
                ┌─────────────────────┐
                │ dataset.           │──► Parquet + MP4 + Metadata
                │ save_episode()     │
                └─────────────────────┘
```

---

## Layer 1: Hardware Communication (Bits on the Wire)

### 1.1 Motor Bus Communication

Motors use **serial communication** via USB-to-TTL adapters. LeRobot supports two motor families:

#### Dynamixel Motors

```python
# Uses dynamixel_sdk for packet-based serial communication
self.port_handler = dxl.PortHandler(self.port)  # e.g., "/dev/ttyUSB0"
self.packet_handler = dxl.PacketHandler(PROTOCOL_VERSION)  # Protocol 2.0
self.sync_reader = dxl.GroupSyncRead(...)  # Efficient batch reads
```

#### Feetech Motors

```python
# Uses scservo_sdk (similar API to dynamixel)
self.port_handler = scs.PortHandler(self.port)
self.packet_handler = scs.PacketHandler(protocol_version)
```

#### Serial Protocol (Byte Level)

```
TX packet (request):  [HEADER] [ID] [LENGTH] [INSTRUCTION] [PARAMS] [CHECKSUM]
RX packet (response): [HEADER] [ID] [LENGTH] [ERROR] [DATA...] [CHECKSUM]
```

Data is raw register values (e.g., position as 12-bit integer 0-4095), then **normalized** to meaningful units:

```python
def sync_read(self, data_name, motors, normalize=True):
    # ... read raw values ...
    if normalize:
        # Convert raw position (0-4095) to radians using calibration
        values = self._normalize_data(data_name, values)
    return values
```

### 1.2 Camera Capture

#### OpenCV Camera

```python
def read(self, color_mode=None):
    start_time = time.perf_counter()
    ret, frame = self.videocapture.read()  # Blocks until frame arrives
    
    if not ret or frame is None:
        raise RuntimeError("Read failed")
    
    processed_frame = self._postprocess_image(frame, color_mode)  # BGR→RGB
    return processed_frame  # numpy.ndarray shape (H, W, 3), dtype uint8
```

#### RealSense Camera

```python
def read(self, color_mode=None, timeout_ms=200):
    ret, frame = self.rs_pipeline.try_wait_for_frames(timeout_ms=timeout_ms)
    color_frame = frame.get_color_frame()
    color_image_raw = np.asanyarray(color_frame.get_data())  # Zero-copy from USB
    return self._postprocess_image(color_image_raw, color_mode)
```

---

## Layer 2: Robot Abstraction

The `Robot` base class (`src/lerobot/robots/robot.py`) defines a unified interface:

```python
class Robot(abc.ABC):
    @abc.abstractmethod
    def get_observation(self) -> dict[str, Any]:
        """Returns flat dict: {"motor1.pos": float, "cam1": np.ndarray, ...}"""
        pass
    
    @abc.abstractmethod
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Sends goal positions to motors, returns actually sent values"""
        pass
```

### Example Implementation (SO100)

```python
def get_observation(self) -> dict[str, Any]:
    # Read all motor positions in one batch (efficient)
    obs_dict = self.bus.sync_read("Present_Position")
    obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
    
    # Capture images from all cameras
    for cam_key, cam in self.cameras.items():
        obs_dict[cam_key] = cam.async_read()  # numpy array (H, W, 3)
    
    return obs_dict  # Flat dict with ~10 motor values + 2-3 images
```

---

## Layer 3: Recording Loop

The main recording loop (`src/lerobot/scripts/lerobot_record.py`) runs at **fixed FPS**:

```python
def record_loop(robot, dataset, fps, teleop, ...):
    timestamp = 0
    start_episode_t = time.perf_counter()
    
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()
        
        # 1. Get current robot state (motors + cameras)
        obs = robot.get_observation()
        
        # 2. Get action from teleoperator (leader arm / keyboard)
        raw_action = teleop.get_action()
        
        # 3. Send action to robot (follower arm)
        _sent_action = robot.send_action(robot_action_to_send)
        
        # 4. Write frame to dataset
        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs, prefix="observation")
            action_frame = build_dataset_frame(dataset.features, raw_action, prefix="action")
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)
        
        # 5. Sleep to maintain target FPS
        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(1 / fps - dt_s)  # e.g., sleep ~30ms for 30 FPS
        
        timestamp = time.perf_counter() - start_episode_t
```

---

## Layer 4: In-Memory Buffering (`add_frame`)

When `dataset.add_frame(frame)` is called:

```python
def add_frame(self, frame: dict) -> None:
    """
    Adds frame to episode_buffer. Images are written to temp PNG files immediately.
    Nothing else is written to disk until save_episode() is called.
    """
    
    # Auto-generate frame_index and timestamp
    frame_index = self.episode_buffer["size"]
    timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
    # ⚠️ NOTE: timestamp = frame_index / fps
    #    This is EPISODE-RELATIVE time, starting from 0 each episode
    
    self.episode_buffer["frame_index"].append(frame_index)
    self.episode_buffer["timestamp"].append(timestamp)
    
    for key in frame:
        if self.features[key]["dtype"] in ["image", "video"]:
            # Write image to temporary PNG immediately
            img_path = self._get_image_file_path(episode_index, image_key=key, frame_index=frame_index)
            self._save_image(frame[key], img_path)
            self.episode_buffer[key].append(str(img_path))
        else:
            # Scalar/vector data stays in memory
            self.episode_buffer[key].append(frame[key])
```

### Temporary Image Storage

```
dataset_root/
└── images/
    └── observation.images.cam1/
        └── episode-000000/
            ├── frame-000000.png
            ├── frame-000001.png
            └── frame-000002.png
```

---

## Layer 5: Episode Finalization (`save_episode`)

When the episode ends:

```python
def save_episode(self, episode_data=None, parallel_encoding=True):
    episode_index = self.episode_buffer["episode_index"]
    episode_length = self.episode_buffer["size"]
    
    # 1. Save tabular data (state, action, timestamps) to Parquet
    ep_metadata = self._save_episode_data(self.episode_buffer)
    
    # 2. Encode video from temporary PNGs → MP4
    if has_video_keys:
        for video_key in self.meta.video_keys:
            ep_metadata.update(self._save_episode_video(video_key, episode_index))
    
    # 3. Save episode metadata (length, timestamps, file locations)
    self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, ep_metadata)
    
    # 4. Clear the buffer
    self.episode_buffer = self.create_episode_buffer()
```

### Video Encoding (PNGs → MP4)

```python
def encode_video_frames(imgs_dir, video_path, fps, vcodec="libsvtav1", g=2, crf=30):
    """
    Args:
        g: GOP size (keyframe interval). 
           g=2 means every 2nd frame is a keyframe.
           g=1 means every frame is a keyframe (best for random access)
    """
    input_list = sorted(glob.glob(str(imgs_dir / "frame-*.png")))
    
    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(vcodec, fps, options={"g": str(g), "crf": str(crf)})
        
        for input_data in input_list:
            with Image.open(input_data) as input_image:
                input_frame = av.VideoFrame.from_image(input_image.convert("RGB"))
                packet = output_stream.encode(input_frame)
                if packet:
                    output.mux(packet)
```

---

## Layer 6: Final Dataset Structure on Disk

```
dataset_root/
├── meta/
│   ├── info.json                           # Dataset metadata
│   ├── tasks.parquet                       # Task descriptions
│   ├── stats.safetensors                   # Feature statistics (mean/std)
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet            # Episode metadata (per-episode)
│
├── data/
│   └── chunk-000/
│       └── file-000.parquet                # Frame-level tabular data
│
└── videos/
    ├── observation.images.cam1/
    │   └── chunk-000/
    │       └── file-000.mp4                # Video file (may contain multiple episodes)
    └── observation.images.cam2/
        └── chunk-000/
            └── file-000.mp4
```

### `meta/info.json`

```json
{
    "codebase_version": "v3.0",
    "robot_type": "so100",
    "fps": 30,
    "total_episodes": 50,
    "total_frames": 15000,
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    "features": {
        "observation.state": {
            "dtype": "float32",
            "shape": [6],
            "names": ["shoulder.pos", "elbow.pos", "wrist_pitch.pos", ...]
        },
        "observation.images.cam1": {
            "dtype": "video",
            "shape": [480, 640, 3]
        },
        "action": {
            "dtype": "float32",
            "shape": [6],
            "names": ["shoulder.pos", "elbow.pos", ...]
        }
    }
}
```

### Episode Metadata (Parquet)

| episode_index | length | dataset_from_index | dataset_to_index | videos/cam1/from_timestamp | videos/cam1/to_timestamp |
|--------------|--------|-------------------|------------------|---------------------------|--------------------------|
| 0            | 300    | 0                 | 300              | 0.0                       | 10.0                     |
| 1            | 280    | 300               | 580              | 10.0                      | 19.33                    |
| 2            | 310    | 580               | 890              | 19.33                     | 29.67                    |

> ⚠️ **Critical:** `from_timestamp` and `to_timestamp` define where each episode sits **within the MP4 file**. If `to_timestamp` exceeds the actual MP4 duration, video decoding will fail!

### Frame-Level Data (Parquet)

| index | episode_index | frame_index | timestamp | observation.state | action |
|-------|--------------|-------------|-----------|-------------------|--------|
| 0     | 0            | 0           | 0.000     | [0.1, 0.2, ...]   | [0.1, 0.2, ...] |
| 1     | 0            | 1           | 0.033     | [0.11, 0.21, ...] | [0.11, 0.21, ...] |
| ...   | ...          | ...         | ...       | ...               | ... |

Key fields:
- `index`: Global frame index across entire dataset
- `frame_index`: Frame index **within** the episode (resets to 0 each episode)
- `timestamp`: Time **within** the episode (resets to 0.0 each episode)

---

## Layer 7: Video Timestamp Computation (During Training)

When training loads a frame, it converts episode-relative timestamps to MP4-absolute positions:

```python
def _query_videos(self, query_timestamps: dict, ep_idx: int) -> dict:
    ep = self.meta.episodes[ep_idx]
    
    frames = {}
    for vid_key, query_ts in query_timestamps.items():
        # Get this episode's position within the MP4 file
        from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
        
        # Convert episode-relative timestamps to MP4-absolute timestamps
        shifted_query_ts = [from_timestamp + ts for ts in query_ts]
        
        # Decode video frames at those absolute timestamps
        video_path = self.meta.get_video_file_path(ep_idx, vid_key)
        frames[vid_key] = decode_video_frames(
            video_path, 
            shifted_query_ts, 
            tolerance_s=self.tolerance_s,
            backend=self.video_backend
        )
    
    return frames
```

### Example Calculation

- Episode 2 has `from_timestamp=19.33`, `to_timestamp=29.67`
- Training requests frame at `timestamp=5.0` (within episode)
- Actual MP4 seek position: `19.33 + 5.0 = 24.33` seconds

---

## Summary: Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. CAPTURE (30 Hz loop)                                                 │
│    Motors: USB-Serial → TTL packet → raw int → normalized radians       │
│    Camera: USB → raw buffer → numpy BGR → RGB array (H,W,3)             │
├─────────────────────────────────────────────────────────────────────────┤
│ 2. OBSERVATION DICT                                                     │
│    {"shoulder.pos": 0.123, "cam1": np.array(...), ...}                  │
├─────────────────────────────────────────────────────────────────────────┤
│ 3. FRAME DICT                                                           │
│    {"observation.state": [6 floats], "observation.images.cam1": array}  │
├─────────────────────────────────────────────────────────────────────────┤
│ 4. EPISODE BUFFER (in memory)                                           │
│    Scalars: Python lists                                                │
│    Images: Written to temp PNGs immediately                             │
├─────────────────────────────────────────────────────────────────────────┤
│ 5. SAVE EPISODE                                                         │
│    Scalars → Parquet (columnar, compressed)                             │
│    PNGs → MP4 (h264/svtav1, g=2 keyframes)                              │
│    Metadata → Episode parquet (from_ts, to_ts, chunk/file indices)      │
├─────────────────────────────────────────────────────────────────────────┤
│ 6. TRAINING LOAD                                                        │
│    Parquet → PyArrow → numpy/torch tensors                              │
│    MP4 → torchcodec/pyav → frame at (from_ts + episode_ts)              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Issue: `to_timestamp > video_duration`

**Symptom:** `IndexError: Invalid frame index` or timestamp tolerance assertion failures.

**Cause:** Episode metadata claims frames exist beyond the actual MP4 file length.

**Fix strategies:**
1. Record with `g=1` (every frame is keyframe) for accurate seeks
2. Verify `to_timestamp ≤ actual_video_duration` in QA notebook
3. Use frame clamping in `decode_video_frames_torchcodec()`

### Issue: Large keyframe gaps (pyav tolerance failures)

**Symptom:** `AssertionError: timestamps violate tolerance (1.7s > 0.0001s)`

**Cause:** The `pyav` backend only seeks to keyframes, which may be far apart.

**Fix:** Use `torchcodec` or `video_reader` backend, or relax tolerance for pyav.

### Issue: Frame drops during recording

**Symptom:** Fewer frames in MP4 than episode metadata expects.

**Cause:** I/O bottleneck during image writing, especially with multiple cameras.

**Fix:** 
- Increase `image_writer_processes` or `image_writer_threads`
- Use faster storage (SSD vs HDD)
- Lower camera resolution during recording

---

## References

- **LeRobot GitHub:** https://github.com/huggingface/lerobot
- **Dataset Format Docs:** `docs/source/lerobot-dataset-v3.mdx`
- **Video Utils:** `src/lerobot/datasets/video_utils.py`
- **Recording Script:** `src/lerobot/scripts/lerobot_record.py`

---

*Generated from LeRobot codebase analysis — December 2024*

