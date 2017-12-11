This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

### Members of Powerthrough Team

* [Mikkel Fly Kragh](https://github.com/mikkelkh) - mikkelkragh@gmail.com
* [Philippe Weingertner](https://github.com/PhilippeW83440) - philippe.weingertner@gmail.com
* [Peter Christiansen](https://github.com/PeteHeine) - repetepc@gmail.com
* [Oleg Potkin](https://github.com/olpotkin) - olpotkin@gmail.com
* [Sebastian Stümper](https://github.com/sstuemper) - sebastian.stuemper@gmail.com

<p align="center">
     <img src="./imgs/rosbag-play.gif" alt="pipeline" width="100%" height="100%">
     <br>rosbag-play.gif
</p>

### Carla Self-Driving Car

Vehicle: 2016 Lincoln MKZ.  
Sensors: 2 Velodyne VLP-16 LiDARs, 1 Delphi radar, 3 FLIR (Point Grey) Blackfly cameras, Xsens IMU.   

<p align="center">
     <img src="./imgs/Carla.jpg" alt="pipeline" width="50%" height="50%">
     <br>Carla.jpg
</p>

Udacity Self-Driving Car Harware Specs
- 31.4 GiB Memory
- Intel Core i7-6700K CPU @ 4 GHz x 8
- GPU TITAN X
- 64-bit OS Ubuntu 16.04 with ROS Kinetic


<p align="center">
     <img src="./imgs/Carla2.jpg" alt="pipeline" width="50%" height="50%">
     <br>Carla2.jpg
</p>

https://medium.com/udacity/how-the-udacity-self-driving-car-works-575365270a40

### Architecture Diagram

![image](imgs/final-project-ros-graph-v2.png)

### Traffic Light Detection Node
<p align="center">
     <img src="./imgs/tl-detector.png" alt="pipeline" width="75%" height="75%">
     <br>tl-detector.png
</p>


<p float="left">
  <img src="/imgs/image72.png" width="45%" /> 
  <img src="/imgs/image71.png" width="45%" />
</p>

### Waypoint Loader Node

- From Autoware

### Waypoint Updater Node

<p align="center">
     <img src="./imgs/waypoint-updater.png" alt="pipeline" width="75%" height="75%">
     <br>waypoint-updater.png
</p>

### Waypoint Follower Node

- Pure Pursuit from Autoware

### Drive By Wire Node

<p align="center">
     <img src="./imgs/dbw.png" alt="pipeline" width="75%" height="75%">
     <br>dbw.png
</p>

### Rosbag Tests Howto

Some notes on testing and checking results with a rosbag file:  
  
1) Download https://drive.google.com/open?id=0B2_h37bMVw3iT0ZEdlF4N01QbHc4  
   you get a file: udacity_succesful_light_detection.bag  
2) Rename the file to dbw_test.rosbag.bag  
3) Move the dbw_test.rosbag.bag file to the project root’s data folder, e.g. CarND-Capstone/data/dbw_test.rosbag.bag  
4) Run dbw_test: roslaunch src/twist_controller/launch/dbw_test.launch  
5) Launch rviz in another terminal just to visualize camera and lidar sensors   
When test is done:  
6) Compare your results to the dbw_test output logged in ros/src/twist_controller in three files: brakes.csv, steers.csv, throttles.csv  

  
Carla vs simulation specificities:
We are at very low speed: < 20 km/h  
1) Throttle is at max 0.025 (instead of 1 potentially at regular speeds)  
2) Brake is not in Nm (mass * accel * wheel_radius) but in (accel * wheel_radius) apparently ...   
   Use a percentage mode: percentage vs mass  
   cf: https://github.com/udacity/sdc-issue-reports/issues/1204  
3) Steer is (or can be just computed as) proposed_angular_velocity (as sent by pure pursuit) * steer_ratio ... that is all ...
   It results in more agressive steering which is OK and even usefull at such low speed  
With these modifications we have a very good match match while running dbw_test.py on dbw_test.rosbag.bag  
  
TL test howto:  
1) roslaunch launch/site.launch  
2) rosbag play udacity_succesful_light_detection.bag  
3) check all images in: ros/src/tl_detector/light_classification/debug/imageXXX.png  
  
GPU WARNING:   
- Carla is using a powerfull GPU Titan X
- Code was tested with a GTX 1080 TI, GTX 980 TI and a GTX 1050
- Faster_rcnn is used for TL detection every 333 ms: so the GPU should run faster_rcnn on an image in less than 333 ms 
  (minus time for other nodes processing and minus time for simulator graphics if you are using it ...)

### Conclusion and Next Steps

- Almost no brake command outside deceleration phases
- MPC support
- Profiling
- Traffic Light Detection optimization

### References

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases/tag/v1.3).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view?usp=sharing) that was recorded on the Udacity self-driving car (a bag demonstraing the correct predictions in autonomous mode can be found [here](https://drive.google.com/open?id=0B2_h37bMVw3iT0ZEdlF4N01QbHc))
2. Unzip the file
```bash
unzip traffic_light_bag_files.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_files/loop_with_traffic_light.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
