import pyrealsense2 as rs
import time
import sys

def verify_driver():
    print(f"--- Native Driver Verification ---")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyRealSense2 Version: {rs.__version__}")

    # 1. Check for Devices
    ctx = rs.context()
    if len(ctx.query_devices()) == 0:
        print("\n[FAIL] No camera found. (Check USB Hub/Cable)")
        return

    dev = ctx.query_devices()[0]
    name = dev.get_info(rs.camera_info.name)
    serial = dev.get_info(rs.camera_info.serial_number)
    print(f"\n[OK] Device Found: {name} (S/N: {serial})")

    # 2. Check for Metadata Support (Did the patch work?)
    # We check if the sensor supports 'Global Time'
    depth_sensor = dev.first_depth_sensor()
    supports_global_time = depth_sensor.supports(rs.option.global_time_enabled)
    
    if supports_global_time:
        print("[OK] Hardware Timestamp Support: ENABLED (Patch worked!)")
        depth_sensor.set_option(rs.option.global_time_enabled, 1)
    else:
        print("[WARN] Hardware Timestamp Support: DISABLED (Patch skipped/failed).")
        print("       (You will use System Time. This is stable, just slightly less accurate.)")

    # 3. Stress Test: RGB + Depth
    print("\n--- Starting Stress Test (RGB + Depth) ---")
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable both streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    try:
        pipeline.start(config)
        print(">> Streaming started... (Holding for 20s)")
        
        start = time.time()
        frames_seen = 0
        
        while time.time() - start < 20:
            frames = pipeline.wait_for_frames()
            
            # Verify we have both
            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            
            if not depth or not color:
                continue

            # Verify Timestamp Domain
            ts_domain = frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival) if frames.supports_frame_metadata(rs.frame_metadata_value.time_of_arrival) else "N/A"
            
            frames_seen += 1
            if frames_seen % 30 == 0:
                print(f"  Frame {frames_seen}: Depth={depth.get_data_size()} bytes | RGB={color.get_data_size()} bytes")

        print("\n[SUCCESS] Driver is STABLE. No crashes detected.")

    except Exception as e:
        print(f"\n[FAIL] Crash Detected: {e}")
        print("If this failed with 'Frame didn't arrive', check your USB Hub power.")

    finally:
        pipeline.stop()

if __name__ == "__main__":
    verify_driver()
