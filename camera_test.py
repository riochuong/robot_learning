#!/usr/bin/env python3
import argparse
import cv2


# --- Detected Cameras ---
# Camera #0:
#   Name: OpenCV Camera @ /dev/video0
#   Type: OpenCV
#   Id: /dev/video0
#   Backend api: V4L2
#   Default stream profile:
#     Format: 0.0
#     Fourcc: YUYV
#     Width: 640
#     Height: 480
#     Fps: 30.0
# --------------------
# Camera #1:
#   Name: OpenCV Camera @ /dev/video2
#   Type: OpenCV
#   Id: /dev/video2
#   Backend api: V4L2
#   Default stream profile:
#     Format: 0.0
#     Fourcc: YUYV
#     Width: 640
#     Height: 480
#     Fps: 30.0
# --------------------
# Camera #2:
#   Name: OpenCV Camera @ /dev/video4
#   Type: OpenCV
#   Id: /dev/video4
#   Backend api: V4L2
#   Default stream profile:
#     Format: 0.0
#     Fourcc: YUYV
#     Width: 640
#     Height: 480
#     Fps: 30.0
# --------------------

# run lerobot-find-cameras to find the camera indices

def parse_args():
    parser = argparse.ArgumentParser(
        description="View multiple /dev/video<N> cameras with OpenCV"
    )
    parser.add_argument(
        "indices",
        type=int,
        nargs="+",
        help="Camera indices N for /dev/videoN (e.g. 0 2 4)"
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()

def open_cameras(indices, width, height, fps):
    caps = []
    for idx in indices:
        dev_path = f"/dev/video{idx}"
        print(f"Opening {dev_path}")
        cap = cv2.VideoCapture(dev_path)
        if not cap.isOpened():
            print(f"Warning: cannot open {dev_path}")
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        caps.append((dev_path, cap))
    return caps

def main():
    args = parse_args()
    cams = open_cameras(args.indices, args.width, args.height, args.fps)
    if not cams:
        print("No cameras opened, exiting")
        return

    while True:
        for dev_path, cap in cams:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to grab frame from {dev_path}")
                continue
            cv2.imshow(f"Camera {dev_path}", frame)

        # single key handler for all windows
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for _, cap in cams:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()