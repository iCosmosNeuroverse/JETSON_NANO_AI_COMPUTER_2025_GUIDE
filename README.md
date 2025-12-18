


# JETSON_NANO_AI_COMPUTER_2025_GUIDE & CODE FOR SELF DRIVING (TEST WITHOUT SERVOS ATTACHED TO NANO)

Side note: This nano unit I cherish. I have had this Jetson nano unit for about 7 years now. It's the same unit that ended up getting me featured by Nvidia for pothole detection project that I had posted. NVIDIA still remains world's most powerful AI driver a company worth over 3 trillion. Go to [Nvidia Jetson feature page](https://developer.nvidia.com/embedded/community/jetson-projects) and search for "pothole" then select smart pothole detector project that pops up in the results, or check out the [short link](https://developer.nvidia.com/embedded/community/jetson-projects/ai_pothole_detector).


This will be a part of my self driving big/full scale hypercar project, for [icognium](https://icognium.github.io/). This same guide can be used for small scale RC projects.

## 3d model (I did modeled in Blender and Rhino3d):

![alt text](https://github.com/iCosmosNeuroverse/JETSON_NANO_AI_COMPUTER_2025_GUIDE_AND_CODE/blob/main/_DIO_LUCCIOLA_3D.png)

## Hand Crafted real, full scale hypercar body and chassis. 

~~> Don't mind the blue wrap/rough exterior, all will a white lacebark inspired material, like seen on the roof:

![alt text](https://github.com/iCosmosNeuroverse/JETSON_NANO_AI_COMPUTER_2025_GUIDE_AND_CODE/blob/main/LUCCIOLA_FULL_SCALE_GIF_1.gif)

.

![alt text](https://github.com/iCosmosNeuroverse/JETSON_NANO_AI_COMPUTER_2025_GUIDE_AND_CODE/blob/main/LUCCIOLA_FULL_SCALE_GIF_2.gif)

.

![alt text](https://github.com/iCosmosNeuroverse/JETSON_NANO_AI_COMPUTER_2025_GUIDE_AND_CODE/blob/main/LUCCIOLA_FULL_SCALE_GIF_3.gif)

Dio Lucciola, my handcrafted hypercar prototype, with steel assembly hubs, rims, control arms and axels, and metal cage structure, but wooden floor; symbolizing land of wood and water, and weight savings for this initial prototype instead of traditional heavier metal floor, and hommage to Joe Harmon's fully fiberglass [supercar 2015, "Splinter"](https://www.reddit.com/r/cars/comments/3vgvs6/a_wooden_supercar_named_splinter_has_been_made/).


# Guide & Code to get Jetson Nano B01 4GB up and running Self Driving using Android Phone as camera (incase your CSI camera port is damaged), or via the same CSI Camera if on board camera is okay. with Jetson nano Ai computer unit as the brain

Use command below to see how much ram your system has. If like mine, your nano has 4 usb ports, 1 CSI camera port, a power jack and micro usb power jack, you have the B01 4GB nano ai computer kit with 4gb. If you have 2 camera ports, you have B02.

```
cat /proc/meminfo | head
```

_Side note: In case you bricked your jetson nano somehow via software, however particularly without physical damage, [follow my little guide to unbrick it](https://www.instagram.com/p/DSGw4dyDi2q/)._


# ON A WINDOWS HOST OR LINUX HOST MACHINE:

1. Download FEI's donkey car/jetcar SD card [prebuilt image](https://peter115342.github.io/FEI_jetracer/docs/FEIcar/FEIcar_installation/) with all dependencies. This repo doesn't immediately provide self driving, as it showcases an object detection demo driven by a human controlling a jetson nano enabled rc car. However, the environment is crucially well established to facilitate self driving, the same I will eventually apply to my full scale hypercar prototype.
2. Remove sd card from nano.
3. Use adaptor, plug in host computer, then format sd card with sd card formatter.4. Burn image with Balena Echer.


# ON NANO:


4. [Install swap file](https://jkjung-avt.github.io/setting-up-nano/) to nano to avoid memory issues. An example of memory issue is your on board CSI camera working once, but failing afterwords. Note 8gb file in /mnt/. Also note your total free space by observing Ubuntu's "Other locations" folder and "Computer", which are virtual locations to look on data about your root folder size.

Confirm swap with 

```
swapon --show
```

or

```
free -h 
```

If you don't make the swap permanent as the tutorial above shows you only need to run swapon after reboot.

```
sudo swapon /mnt/8GB.swap
```

Other helpful commands

```
sudo nano /etc/fstab  (before entering /etc/fstab to add entry
...enter file and add entry as seen in tutorial...

sudo chmod 600 /mnt/8GB.swap --secure swap after mk command


sudo swapoff /mnt/8GB.swap  --turn off swap
sudo swapon -a  --turn on swap
swapon --show  --verify on
```

So a straightforward route looks like:

```
mk command to make swap file > chmod to secure swap file > make it permanent if desired (or just run swapon command with the file name upon reboot) > show command to verify swap
```


5. Time to run some demos. Open terminal and run commands:

```
>>> source env/bin/activate
```

Note we won't use Fei's manage.py mod because it includes yolo detection and has no self driving pilot prepared. Feis environment overall, i.e. the prebuilt sd card image, is ideal but the manage.py file is not configured for self driving, only experimental object recognition by human driven nano unit/car.

You will instead create a new donkeycar app with command below then navigate to the folder. 

```
donkey createcar --path ~/<carprojectname>
```

Example:

```
donkey createcar --path ~/jamaica_ai_car
```

# 6. Copy my python files manage.py, manage_ByIpCam.py, csi_cam_test.py, and  ip_cam_test.py to the "jamaica_ai_car" or your directory if you made one by another name.

Note, before invoking drive on the nano, if you just have a camera and no servo /steering/throttle pwm yet attached to NANO, you can do a dev test by disabling the following /home/jamaica_ai_car/myconfig.py. Add the following lines to the top of the file:

```
SHOW_PILOT_STATS = True
USE_CONSTANT_THROTTLE = True
THROTTLE = 0.25
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

You can also test ip cam by invoking my csi_cam_test.py file. >>> python csi_cam_test.py

7. Time to invoke self driving inference:
   
[Acquire donkey pretrained models](https://github.com/autorope/donkey_datasets) from circuit_launch_20210716/models and place them in jamaica_ai_car/models folder. Integrate when asked. 

Use pretrained pilot ai for self driving inference with external ipcamera via phone from playstore like IPWebcam on android  or onboard CSICamera

By now you would have copied manage.py file to the new donkeycar folder you just created or jamaica_ai_car if you used my name example.
Also copy csi_cam_test.py, and ip_cam_test.py  to your new project dir. 

use >>> python csi_cam_test.py and >>> python ip_cam_test.py to test for csi or ip cam.

# Warning: Close with CTR+C in terminal to close camera feed test window properly. Or cam will fail to start next time.

~~Unstuck cam state if you forgot to use CTRL+C to close last argus camera to avoid position/camera error next cam test attempt

```
>>> sudo systemctl restart nvargus-daemon
```
#Also, crucially, copy keras.py to a FEIcar/donkeycar/parts/ and replaced when asked. _I modified donkeycar file to accommodate for missing CSI camera frames that cause crash during stream. manage.py was also modified to accommodate as well as do things like add IPcamera support, and crucially enable loading of pretrained model for self driving. I used the onboard CSI nano camera, but coded the IPcamera as an alternative to enable external android phone cam to be used as camera source._

8. Navigate:

```
cd <your new donkeycar project name>
```
Example
```
cd jamaica_ai_car
```

#9. Invoke manage.py with a trt model which is lightweight and faster than the h5 models from the dataset zoo.
This is what facilitates self driving!

```
>>> python manage.py drive --model "models/pilot_21-08-12_4.trt" --throttle 0.25 (uses csi onboard instead of external ip android cam)
```

Instead of onboard camera, use IP Android Camera via app like IPWebcam then hit start server button buttom options after scrolling down:

```
>>> python manage.py drive --ip_cam_url <android cam ip>" --model "<pretrained pilot driver model>" --throttle 0.25 (uses IPCamera/external android cam)
```

Example
```
>>> python manage_ByIpCam.py --ip_cam_url http://192.168.100.192:8080/video" --model "models/pilot_21-08-12_4.trt" --throttle 0.25 (uses IPCamera/external android cam)
```

Wait for this after executing manage.py then go to <jetsonip>:port  on another device connected to internet with a screen you can use. 
You will be connecting to Jetson unit that's basically now a donkey car instance.

```
Recording Change = False
Setting Recording = False
```

For eg you will navigate to  192.168.100.197:8087 on your Android phone browser for eg and there you will see your live Jetson nano I computer donkeycar instance. 
Here you have the option of driving the nano as a rc car to collect images and write them to tubs for training granted if it was connected to wheels and servos or enable it to drive itself based on a loaded pilot.

In my case I test here before connecting nano to my full scale real life hypercar prototype. In other caes people use this to test before connecting to their small scale rc car.

#Below is what the screen on the phone or computer will look like when you access the live nano instance.

#My nano is running browser and also showing the live stream of what the csi mpi camera is seeing, which is the screen itself that nano is connected to:

![alt text](https://github.com/iCosmosNeuroverse/JETSON_NANO_AI_COMPUTER_2025_GUIDE_AND_CODE/blob/main/cosmos_jetson_nano_ip_connection_capture_2.png)

 .
 .
 
![alt text](https://github.com/iCosmosNeuroverse/JETSON_NANO_AI_COMPUTER_2025_GUIDE_AND_CODE/blob/main/cosmos_jetson_nano_ip_connection_capture.png)


# An example road clip you can use to "partially simulate" the real jetson unit travelling in reality while using a virtual screen of road being traversed, before it's camera, by focusing camera on a computer screen playing the clip. 

You can use nano itself to run the video while focusing it's own camera on the clip though that could be costly memory wise. Maybe download the clip 240 to 360p and run it as a native video instead of on chrome instance.

This way one may call "Screen-based perception testing" or "Vision in the loop" testing where nano is seeing the road and responding by sending out it's output signals though not to PWM modules to servos and wheels.

Useful before attaching wheels and steering to nano.

[https://youtu.be/Nhg4BjgkWGQ](https://youtu.be/Nhg4BjgkWGQ)




