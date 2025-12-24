


# JETSON_NANO_AI_COMPUTER_2025_GUIDE & CODE FOR SELF DRIVING (TEST WITHOUT SERVOS ATTACHED TO NANO)

Side note: This nano unit I cherish. I have had this Jetson nano unit for about 7 years now. It's the same unit that ended up getting me featured by Nvidia for pothole detection project that I had posted. NVIDIA still remains world's most powerful AI driver a company worth over 3 trillion. Go to [Nvidia Jetson feature page](https://developer.nvidia.com/embedded/community/jetson-projects) and search for "pothole" then select smart pothole detector project that pops up in the results, or check out the [short link](https://developer.nvidia.com/embedded/community/jetson-projects/ai_pothole_detector).


This will be a part of my self driving big/full scale hypercar project, for [icognium](https://icognium.github.io/). This same guide can be used for small scale RC projects.

## 3d model (I did modeled in Blender and Rhino3d):

![alt text](https://github.com/iCosmosNeuroverse/JETSON_NANO_AI_COMPUTER_2025_GUIDE_AND_CODE/blob/main/_DIO_LUCCIOLA_3D.png)

## I then hand crafted real, [full scale hypercar body and chassis](https://www.youtube.com/watch?v=LPUompopWgU). (It's quite wide, as seen):

~~> Don't mind the blue wrap/rough exterior, all will a white lacebark inspired material, like seen on the roof (note the roof itself will be adjusted too):


![alt text](https://github.com/iCosmosNeuroverse/JETSON_NANO_AI_COMPUTER_2025_GUIDE_AND_CODE/blob/main/LUCCIOLA_FULL_SCALE_GIF_1_GIF.gif)
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

1. Download FEI's donkey car/jetcar SD card [prebuilt image](https://peter115342.github.io/FEI_jetracer/docs/FEIcar/FEIcar_installation/) with all dependencies. This repo doesn't immediately provide self driving, as it showcases an object detection demo driven by a human controlling a jetson nano enabled rc car. However, the environment is crucially well established (except for some crucial modifications) to facilitate self driving, the same I will eventually apply to my full scale hypercar prototype.
2. Remove sd card from nano.
3. Use adaptor, plug in host computer, then format sd card with sd card formatter.
4. Burn image with Balena Echer.


# ON NANO:


5. [Install swap file](https://jkjung-avt.github.io/setting-up-nano/) to nano to avoid memory issues. An example of memory issue is your on board CSI camera working once, but failing afterwords. Note 8gb file in /mnt/. Also note your total free space by observing Ubuntu's "Other locations" folder and "Computer", which are virtual locations to look on data about your root folder size.

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


6. Time to run some demos. Open terminal and run commands:

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

# 7. Copy my python files manage.py, manage_ByIpCam.py, csi_cam_test.py, and  ip_cam_test.py to the "jamaica_ai_car" or your directory if you made one by another name. 

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
ROI_CROP_TOP = 45 (DELETE THIS COMMENT IN BRACKETS BEFORE PASTING!: If you use a Crop based categorical model, use this otherwise remove as guided by donkey pretrained page)
```

Before next command, TEST YOUR NANO CAMERA FIRST, a window should pop up showing feed based on your onboard jetson camera:

```
>>> gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1' ! nvvidconv ! 'video/x-raw, format=BGRx' ! videoconvert ! xvimagesink
```

You can also test ip cam by invoking my csi_cam_test.py file. >>> python csi_cam_test.py

8. Time to invoke self driving inference:
   
[Acquire donkey pretrained models](https://github.com/autorope/donkey_datasets) from circuit_launch_20210716/models and place them in jamaica_ai_car/models folder. Integrate when asked. 

Note: Pay close attention to the [models table](https://github.com/autorope/donkey_datasets/tree/master/circuit_launch_20210716#the-pilot-table ) on the donkeycar datasets /pretrained models page. 
If the model you downloaded ends with 17, then adjust manage.py's pilot loading code based on the table, i.e. 17 corresponds to a categorical model hence KerasCatgeorical I used in the code.

Note: that if you try to use trt models, you'd use something like TRTCategorical or linear depending on table, but TRT is not installed in the virtual environment loaded overall, so try to stick to categorical models or keras aligned models.


# Use pretrained pilot ai for self driving inference with external ipcamera via phone from playstore like IPWebcam on android  or onboard CSICamera

By now you would have copied manage.py file to the new donkeycar folder you just created or jamaica_ai_car if you used my name example.
Also copy csi_cam_test.py, and ip_cam_test.py  to your new project dir. 

use >>> python csi_cam_test.py and >>> python ip_cam_test.py to test for csi or ip cam.

# Warning: Close with CTR+C in terminal to close camera feed test window properly. Or cam will fail to start next time.

~~Unstuck cam state if you forgot to use CTRL+C to close last argus camera to avoid position/camera error next cam test attempt

```
>>> sudo systemctl restart nvargus-daemon
```
#Also, crucially, copy keras.py to a FEIcar/donkeycar/parts/ and replaced when asked. _I modified this donkeycar file to accommodate for missing CSI camera frames that cause crash during stream. manage.py was also modified to accommodate as well as do things like add IPcamera support, and crucially enable loading of pretrained model for self driving. I used the onboard CSI nano camera, but coded the IPcamera as an alternative to enable external android phone cam to be used as camera source._

#Also, crucially once more, copy interpreter.py to /Home/FEICar/donkeycar/parts/interepreter.py and replaced when asked. _I modified this donkeycar file to purge a numpy issue related to pretrained model loading.

9. Navigate:

```
cd <your new donkeycar project name>
```
Example
```
cd jamaica_ai_car
```

#10. Invoke manage.py with a h5 model from the dataset zoo.
This is what facilitates self driving!

```
>>> python manage.py drive --model "models/pilot_21-08-13_17.h5"  (uses csi onboard instead of external ip android cam)
```

Instead of onboard camera, use IP Android Camera via app like IPWebcam then hit start server button buttom options after scrolling down:

```
>>> python manage.py drive --ip_cam_url <android cam ip>" --model "<pretrained pilot driver model>"  (uses IPCamera/external android cam)
```

Example
```
>>> python manage_ByIpCam.py --ip_cam_url http://192.168.100.192:8080/video" --model "models/pilot_21-08-13_17.h5" (uses IPCamera/external android cam)
```

Wait for this after executing manage.py then go to <jetsonip>:port  on another device connected to internet with a screen you can use. 
You will be connecting to Jetson unit that's basically now a donkey car instance.

```
Recording Change = False
Setting Recording = False
```

For eg you will navigate to something looking like 192.168.100.197:8087 on your Android phone browser for eg and there you will see your live Jetson nano AI computer donkeycar instance. (run >>> ifconfig on your nano, and grab the ip. Don't grab the broadcast ip) 

Here you have the option of driving the nano as a rc car to collect images and write them to tubs for training granted if it was connected to wheels and servos or enable it to drive itself based on a loaded pilot.

In my case I test here before connecting nano to my full scale real life hypercar prototype. In other caes people use this to test before connecting to their small scale rc car.

#Below is what the screen on the phone or computer will look like when you access the live nano instance.

#My nano is running browser and also showing the live stream of what the csi mpi camera is seeing, which is the screen itself that nano is connected to:

![alt text](https://github.com/iCosmosNeuroverse/JETSON_NANO_AI_COMPUTER_2025_GUIDE_AND_CODE/blob/main/cosmos_jetson_nano_ip_connection_capture_2.png)

---Note: When you connect to jetson nano unit using another device with internet access, once you switch from User Mode to Auto Steer notice how laggy the camera feed will become. This is 
a clear sign that nano has began to do inference based on the loaded .h5 model. 


 .
 .
 
![alt text](https://github.com/iCosmosNeuroverse/JETSON_NANO_AI_COMPUTER_2025_GUIDE_AND_CODE/blob/main/cosmos_jetson_nano_ip_connection_capture.png)

.
.

![alt text](https://github.com/iCosmosNeuroverse/JETSON_NANO_AI_COMPUTER_2025_GUIDE_AND_CODE/blob/main/cosmos_jetson_nano_ip_connection_capture_3.png)

## I used auto steer mode above, enabling steering, while throttle is fixed. If you correctly replaced the files, and also loaded the appropriate model wrt the donkey pretrained guide table, against the Module specified in manage.py you will successfully transition from user to auto steer mode without model mismatch/loading or numpy errors.



Note: Do not hit "Start vehicle button on ui" as the vehicle should laready be running given "starting vehicle at <value> hz" in terminal.

Note: If you don't see a camera feed in the ui, or see an error, please ensure the prior steps especially copying my keras.py file, and manage.py file was done appropiately.

# If process closed improperly check if port is still open, to avoid OSAddress usage error.

```
>>> sudo netstat -tulpn | grep 8887
```

kill all processes using the port

```
>>> sudo fuser -k 8887/tcp
```


# An example road clip you can use to "partially simulate" the real jetson unit travelling in reality while using a virtual screen of road being traversed, before it's camera, by focusing camera on a computer screen playing the clip. 

You can use nano itself to run the video while focusing it's own camera on the clip though that could be costly memory wise. Maybe download the clip 240 to 360p and run it as a native video instead of on chrome instance.

This is what some may call "Screen-based perception testing" or "Vision in the loop" testing where nano is seeing the road and responding by sending out it's output signals though not to PWM modules to servos and wheels.

Useful before attaching wheels and steering to nano.

[https://youtu.be/Nhg4BjgkWGQ](https://youtu.be/Nhg4BjgkWGQ)

# For convenient testing it may be helpful to acquire a long MPI/CSI camera cable, and swap out the short default one that ships with jetson nano kit.

You can buy the cable here: https://www.ebay.com/itm/116151757885

In my case I need my camera wire to be long enough to stretch along some length of the full scale hypercar, so the camera can view the road ahead.

Even before being attached to the car, it's now convenient to do Screen based perception testing; i.e. the nano unit can be off to the side on a table while I can hold the mpi camera comfortably while focusing it on a screen with a road being traversed.


# A quick little hack incase your CSI camera secure latch is damaged and missing:

You can find a medicine cartridge of pills, cut two rectangular pieces of the container which are thin, stack them together, then use tape to wrap creating a handle you can then bend, leaving a portion of the stacked pieces exposed.

Then use those to secure the CSI camera, by touching atleast one set of vertices then gently pushing dowards using the thumb on the handle bend.

![alt text](https://github.com/iCosmosNeuroverse/JETSON_NANO_AI_COMPUTER_2025_GUIDE_AND_CODE/blob/main/cosmos_jetson_nano_camera_latch_hack.png)

# Additional pretrained models to try

Try [muosvr's pretrained models](https://github.com/Muosvr/mycar), and optionally, you can rename the no extention files in model dir.
When investigating the .File files, there's hdf5 in the header, indicating it's indeed h5 files that were probbaly renamed to .File when he uploaded the repo.

Command to analyze file header:

```
Format-Hex model_17.File | Select-Object -First 5
```

Result containing HDf5 data proving h5 origin which donkeycar/tf can supposedly consume:

```
           00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F

00000000   89 48 44 46 0D 0A 1A 0A 00 00 00 00 00 08 08 00  HDF............
00000010   04 00 10 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
00000020   FF FF FF FF FF FF FF FF C8 FE 31 00 00 00 00 00  ........Èþ1.....
00000030   FF FF FF FF FF FF FF FF 00 00 00 00 00 00 00 00  ................
00000040   60 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00  `...............
```


Navigate to models folder in muosvr's downloaded repo, then run command below if you want to rename, though that's optional, as keras typically check content not names:

```
Get-ChildItem -File | Where-Object { $_.Extension -eq "" } | ForEach-Object {
    $newName = $_.Name + ".h5"
    Write-Host "Would rename '$($_.Name)' to '$newName'"
}
```

Test loading in python on your desktop before puting on jetson nano:

```
>>> from tensorflow import keras
>>> model = keras.models.load_model("models/mypilot.h5")
```

If you [click on muosvr's blog link in the repo](https://medium.com/@jasonwu_49390/donkey-car-part-3-neural-net-1f9b7ea939c), you can see he's listed a ranking of the pretrained models. 
The top most entry/first row represents the most accurate model with the least loss.
It represents a model that processes multiple images compareed to the single image default mypilot model, with a bit of performance sacrifice.

![alt text](https://github.com/iCosmosNeuroverse/JETSON_NANO_AI_COMPUTER_2025_GUIDE_AND_CODE/blob/main/muosvr_pretrained_rank.png)


# Test prediction capacity

```
from tensorflow import keras
import numpy as np

# Standard CNN: Single frame
dummy_frame = np.zeros((1,120, 160, 3), dtype=np.float32)
model = keras.models.load_model("models/mypilot")
pred=model.predict(dummy_frame)



# 3DCNN: Stack 3 frames as 3DCNNS need multiple frames
dummy_frames = np.stack([dummy_frame, dummy_frame, dummy_frame], axis=0)

# Add batch dimension
x = np.expand_dims(dummy_frames, axis=0)

model = keras.models.load_model("models/speedup3dcnn/pilot_3Dspeedup")
pred=model.predict(dummy_frames)
```
