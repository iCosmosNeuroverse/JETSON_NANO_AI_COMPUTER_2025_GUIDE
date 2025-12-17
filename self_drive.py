#!/usr/bin/env python3
#Example invoke>>> python self_drive.py "http://192.168.100.192:8080/video" "models/pilot_21-08-12_4.h5" 
#Example invoke>>> python self_drive.py "models/pilot_21-08-12_4.h5" (uses csi onboard instead of external ip android cam)

import time
import cv2
from donkeycar.vehicle import Vehicle
from donkeycar.parts.keras import KerasPilot
from donkeycar.parts.keras import KerasLinear
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

import sys



# =============================
# Main
# =============================


if __name__ == "__main__":

    V = Vehicle()
    if len(sys.argv) > 2:
        ip_cam_url = sys.argv [ 1 ]
        model_path = sys.argv [ 2 ]
        #while camera.run() is None:
        #    print("Preparing ipcamera data...")
        #    time.sleep(0.1)

        # Add camera part
        camera = IPCamera(ip_cam_url, IMAGE_W, IMAGE_H)


        V.add(camera, outputs=['image'], threaded=False)
        # Add ML pilot
        pilot = KerasLinear()
        pilot.load(model_path)
        V.add(pilot, inputs=['image'], outputs=['angle', 'throttle'])

        # Add display
        display = DisplayOutput()
        V.add(display, inputs=['image', 'angle', 'throttle'])

        # Start vehicle loop
        V.start(rate_hz=RATE_HZ)
    


    elif len(sys.argv) <= 2:
        from donkeycar.parts.camera import CSICamera
        camera = CSICamera(480,320, 2,480, 320,15, False)
        V.add(camera, outputs=['image'], threaded=True)

        model_path = sys.argv [ 1 ]
        # Add ML pilot
        pilot = KerasLinear()
        pilot.load(model_path)
        V.add(pilot, inputs=['image'], outputs=['angle', 'throttle'])

        # Add display
        display = DisplayOutput()
        V.add(display, inputs=['image', 'angle', 'throttle'])

        # Start vehicle loop
        V.start(rate_hz=RATE_HZ)
        # Add camera part

