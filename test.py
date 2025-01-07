import cv2

def try_camera(camera_id):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Cannot access webcam with camera ID {camera_id}")
        return False
    else:
        print(f"Webcam accessed successfully with camera ID {camera_id}")
        cap.release()
        return True

# Try different camera IDs
camera_ids = [0, 1, 2, -1]
for camera_id in camera_ids:
    if try_camera(camera_id):
        break
else:
    print("No valid camera ID found.")