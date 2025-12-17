# JETSON_NANO_AI_COMPUTER_2025_GUIDE FOR SELF DRIVING

Side note: This nano unit I cherish. I have had this Jetson nano unit for about 7 years now. It's the same unit that ended up getting me featured by Nvidia for pothole detection project that I had posted. NVIDIA still remains world's most powerful AI driver a company worth over 3 trillion. Go to [Nvidia Jetson feature page](https://developer.nvidia.com/embedded/community/jetson-projects) and search for "pothole" then select smart pothole detector project that pops up in the results, or check out the [short link](https://developer.nvidia.com/embedded/community/jetson-projects/ai_pothole_detector).


# Guide to get Jetson Nano B01 4GB up and running Self Driving using Android Phone as camera (incase your CSI camera port is damaged), or via the same CSI Camera if on board camera is okay. with Jetson nano Ai computer unit as the brain



# ON A WINDOWS HOST OR LINUX HOST MACHINE:

1. Download FEI's donkey car/jetcar SD card [prebuilt image](https://peter115342.github.io/FEI_jetracer/docs/FEIcar/FEIcar_installation/) with all dependencies. This repo doesn't immediately provide self driving, as it showcases an object detection demo driven by a human controlling a jetson nano enabled rc car. However, the environment is crucially well established to facilitate self driving, the same I will eventually apply to my full scale hypercar prototype.
2. Remove sd card from nano.
3. Use adaptor, plug in host computer, then format sd card with sd card formatter.4. Burn image with Balena Echer.


# ON NANO:


4. [Install swap file](https://jkjung-avt.github.io/setting-up-nano/) to nano to avoid memory issues. An example of memory issue is your on board CSI camera working once, but failing afterwords.


5. Time to run some demos. Open terminal and run commands:

```
>>> source env/bin/activate
>>> cd ~/mycar
```

6. Copy my python files self_drive_basic.py, self_drive.py, and csi_cam_test.py, and  ip_cam_test.py to the "mycar" directory.

Note, before invoking drive on the nano, if you just have a camera and no servo /steering/throttle pwm yet attached to NANO, you can do a dev test by disabling the following /home/mycar/myconfig.py. Add the following lines to the top of the file after "YOLO_NMS_THRESHOLD = 0.4" and before the commented out section. Ensure they aren't commented out:

```
DRIVE_TRAIN_TYPE="mock"
CONTROLLER_TYPE="mock"
USE_JOYSTICK_AS_DEFAULT=False
HAVE_IMU = False
USE_SSD1306_128_32 = False
HAVE_ODOM = False
USE_LIDAR = False
HAVE_TFMINI = False
HAVE_RGB_LED = False
```

Before next command, TEST YOUR NANO CAMERA FIRST, a window should pop up showing feed based on your onboard jetson camera:

```
>>> gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1' ! nvvidconv ! 'video/x-raw, format=BGRx' ! videoconvert ! xvimagesink
```

You can talso test ip cam by invoking my csi_cam_test.py file. >>> python csi_cam_test.py

7. Time to invoke object detection using jetson nano:

```
>>> python manage_.py drive 
```

8. FOR USERS WITHOUT CSI CAMERA:

Invoke my modified manage_.py file if you don't have nano CSI camera, and you have android and IPWebcam app. Start app, then start server on phone which shows a camera stream.

You can test ip cam by invoking my ip_cam_test.py file. >>> python ip_cam_test.py

```
>>> python manage.py drive --ip_cam_url="<Your phone ip cam url or other device ip cam url>"
```

Eg:

```
>>> python manage_.py drive --ip_cam_url="http://192.168.100.192:8080/video" (manage_.py was originally from Fei, but I modified and created a separate file "manage_" to have IPCamera usage functionality)
```

[Acquire donkey pretrained models](https://github.com/autorope/donkey_datasets) from circuit_launch_20210716/models and place them in mycar/models folder. Integrate when asked. 

Use pretrained pilot ai for self driving inference with external ipcamera via phone from playstore like IPWebcam on android  or onboard CSICamera

#Example invoke

```
>>> python self_drive.py "http://192.168.100.192:8080/video" "models/pilot_21-08-12_4.h5"
```

#Example invoke

```
>>> python self_drive.py "models/pilot_21-08-12_4.h5" (uses csi onboard instead of external ip android cam)
```


Use non ml for self driving inference with external ipcamera via phone from playstore like IPWebcam on android  or onboard CSICamera

#Example invoke

```
>>> python self_drive_basic.py "http://192.168.100.192:8080/video" "models/pilot_21-08-12_4.h5" (External IP camera android phone etc feed)
```

#Example invoke

```
>>> python self_drive_basic.py "models/pilot_21-08-12_4.h5" (uses csi onboard instead of external ip android cam)
```

Note your nano ip. You can navigate to it on host  computer and drive the nano by ui controls if you had joystick. May be possible to use a separate mobile. Eg my nano ip:

```
http://192.168.100.197:8887
```




