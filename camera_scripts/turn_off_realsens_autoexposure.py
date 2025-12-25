import pyrealsense2 as rs
import time

def force_setup_d435i():
    # Create a context to see all connected devices
    ctx = rs.context()
    devices = ctx.query_devices()
    print(devices)
    from pprint import pprint
    found = False
    for dev in devices:
        print(dev)
        # Check if it is a D400 series camera (D435i, D455, etc.)
        name = dev.get_info(rs.camera_info.name)
        print("Name:", name)
        if "D435" in name or "D400" in name:
            print(f"Configuring {name}...")
            found = True
            
            # The RGB Camera is usually Sensor Index 1
            # (Index 0 is usually the Depth/Stereo sensor)
            sensors = dev.query_sensors()
            for sensor in sensors:
                if sensor.get_info(rs.camera_info.name) == "RGB Camera":
                    # OPTION 1: Disable Auto Exposure Priority
                    # 0.0 = OFF (Forces constant FPS)
                    # 1.0 = ON (Allows FPS drop for better exposure)
                    sensor.set_option(rs.option.auto_exposure_priority, 0.0)
                    print(" - Auto Exposure Priority: DISABLED (FPS Locked)")
                    
                    # OPTION 2: Lock White Balance (Optional but recommended)
                    # sensor.set_option(rs.option.enable_auto_white_balance, 0.0)
                    # sensor.set_option(rs.option.white_balance, 4000)
                    
    if not found:
        print("No RealSense D435i found!")

if __name__ == "__main__":
    force_setup_d435i()
