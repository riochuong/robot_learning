import pyrealsense2 as rs

ctx = rs.context()
dev = ctx.query_devices()[0]
pipeline = rs.pipeline()
pipeline.start()

frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()

# 1. Get the Timestamp
ts = color_frame.get_timestamp()

# 2. Get the "Domain" (Source of Truth)
domain = color_frame.get_frame_timestamp_domain()

print(f"Timestamp: {ts}")
print(f"Domain: {domain}") 
# If it says 'RS2_TIMESTAMP_DOMAIN_SYSTEM_TIME', you are using the Host Clock (Unpatched).
# If it says 'RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK', you are using the Camera Clock (Patched).

pipeline.stop()
