import pyrealsense2 as rs
import time
import cv2
import numpy as np

def test_safe_mode():
    print("--- Starting SAFE MODE (RGB Only, No Depth, No IMU) ---")
    
    # 1. Create a Context
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("No device found! (Did the port die again?)")
        return

    # 2. Configure the Pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Explicitly enable ONLY RGB
    # We intentionally do NOT call enable_stream for DEPTH, GYRO, or ACCEL
    width, height, fps = 640, 480, 30
    print(f"Requesting: RGB {width}x{height} @ {fps} FPS")
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

    # 3. Start Streaming
    try:
        pipeline.start(config)
        print(">> Pipeline started successfully!")
        print(">> Streaming for 20 seconds... Press CTRL+C to stop early.")
        
        start_time = time.time()
        frames_count = 0
        
        while time.time() - start_time < 300:
            # Wait for frames
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
                
            # Convert to numpy for sanity check (simulating LeRobot workload)
            img = np.asanyarray(color_frame.get_data())
            frames_count += 1
            
            if frames_count % 30 == 0:
                print(f"  [Running] Frame {frames_count} - Res: {img.shape}")

        print(">> Success! System remained stable.")
        
    except RuntimeError as e:
        print(f"\n[CRASH] RealSense Error: {e}")
        print("This usually means the USB controller died or the cable failed.")
        
    finally:
        pipeline.stop()
        print("Pipeline stopped.")

if __name__ == "__main__":
    test_safe_mode()
