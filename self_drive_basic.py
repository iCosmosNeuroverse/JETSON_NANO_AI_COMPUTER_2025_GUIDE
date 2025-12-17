#Example invoke>>> python self_drive_basic.py "http://192.168.100.192:8080/video"
#Example invoke>>> python self_drive_basic.py (uses csi onboard instead of external ip android cam)

#!/usr/bin/env python3


import time
import cv2
from donkeycar.vehicle import Vehicle
import numpy as np
from donkeycar.parts.transform import Lambda


# =============================
# Configuration
# =============================
IMAGE_W = 160
IMAGE_H = 120 
RATE_HZ = 20


# =============================
# IP Webcam Camera Part (Threaded)
# =============================
class IPCamera:
    def __init__(self, url, image_w, image_h):
        self.cap = cv2.VideoCapture(url)
        self.image_w = image_w
        self.image_h = image_h
        self.frame = None
        self.running = True

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open IP Webcam stream")

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (self.image_w, self.image_h))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame = frame
            else:
                time.sleep(0.01)

    def run(self):
        return self.frame

    def shutdown(self):
        self.running = False
        self.cap.release()


# ------------------------------------------------
# CONSTANT THROTTLE PART
# ------------------------------------------------
class ConstantThrottle:
    def __init__(self, throttle=0.2):
        self.throttle = throttle

    def run(self):
        return self.throttle


# =============================
# Display Output Part
# =============================
class DisplayOutput:
    def run(self, image, angle, throttle):
        if image is None:
            return

        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.putText(img, f"Angle: {angle:.2f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(img, f"Throttle: {throttle:.2f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("DonkeyCar IP Webcam Inference (Threaded)", img)
        cv2.waitKey(1)



# ------------------------------------------------
# CV Line Follower Part
# ------------------------------------------------
class YellowLineFollower:
    def __init__(self):
        # HSV range for yellow (TUNE THIS)
        self.lower = np.array([20, 100, 100])
        self.upper = np.array([35, 255, 255])

    def run(self, image):
        if image is None:
            return 0.0

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Mask yellow
        mask = cv2.inRange(hsv, self.lower, self.upper)

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return 0.0

        # Largest contour
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)

        if M['m00'] == 0:
            return 0.0

        cx = int(M['m10'] / M['m00'])
        w = image.shape[1]

        # Normalize to [-1, 1]
        error = (cx - w / 2) / (w / 2)

        # Steering angle
        angle = float(np.clip(error, -1.0, 1.0))
        return angle



# =============================
# Main 
# =============================
import sys

if __name__ == "__main__":

    V = Vehicle()

    if len(sys.argv) > 1:
        ip_cam_url = sys.argv [ 1 ]

	# Add camera part
        camera = IPCamera(ip_cam_url, IMAGE_W, IMAGE_H)
        V.add(camera, outputs=['image'], threaded=False)


        V.add(YellowLineFollower(),
        inputs=['image'], outputs=['angle'])

        V.add(ConstantThrottle(0.2),
        outputs=['throttle'])

        V.start(rate_hz=20)

        V.start(rate_hz=20)

    elif len(sys.argv) <=1 :
	# Add camera part
        from donkeycar.parts.camera import CSICamera
        camera = CSICamera(480,320, 2,480, 320,15, False)
        V.add(camera, outputs=['image'], threaded=True)


        V.add(YellowLineFollower(),
        inputs=['image'], outputs=['angle'])

        V.add(ConstantThrottle(0.2),
        outputs=['throttle'])

        V.start(rate_hz=20)
