#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import os
import cv2
import yaml
import time
import numpy as np
from scipy import spatial
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import quaternion

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        
        # Folder of where images are stored. 
        self.dirData = os.path.join("/home/pistol/DataFolder/SelfDriving/TrafficLights")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()
        self.idx = 0
        self.waypointlist = np.array([])
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.tree = []
        self.image_count = 0
        self.waypoints_prepared = False
        self.waypoints_received = False
        self.idx_traffic_light_found = False
        self.update_waypoint = False
        
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        print("WAYPOINTS_RECEIVED")
        self.waypoints = waypoints
        self.waypoints_received = True
        self.update_waypoint = True

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.camera_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.has_image = True
        self.image_count = self.image_count+1
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """

        # The closest waypoint is determiend using a KDtree - just to be difficult.
        useKDTree = True
        if(self.waypoints_prepared == False) or self.update_waypoint:
            # Placing waypoints in list
            self.waypointlist = np.array([[waypoint.pose.pose.position.x,
                                           waypoint.pose.pose.position.y,
                                           waypoint.pose.pose.orientation.x,
                                           waypoint.pose.pose.orientation.y,
                                           waypoint.pose.pose.orientation.z,
                                           waypoint.pose.pose.orientation.w] for waypoint in self.waypoints.waypoints])
            
            # Determining the KDTree to faster measure distance to closest point
            if useKDTree:
                self.tree = spatial.KDTree(self.waypointlist[:,0:2],1)
            self.waypoints_prepared = True
            self.update_waypoint = False
            
        
        # Get closest idx. 
        if(self.waypoints_prepared):
            if useKDTree:
                dist,idxs = self.tree.query(pose)
                self.idx = idxs
            else:
                # Otherwise calculate it
                dist = np.sum(np.power(pose-self.waypointlist[:,0:2],2),axis=1)
                self.idx = np.argmin(dist)
        else:
            self.idx = 0
            dist = 0
            
        #print("GetClosestWayPoint: ", self.idx,dist)
        return self.idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        #Get classification
        return self.light_classifier.get_classification(self.camera_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        idx_waypoint_traffic_light = -1
        
        
        # This is only processed if waypoints have been received. 
        if self.waypoints_received:
            
            # List of positions that correspond to the line to stop in front of for a given intersection
            stop_line_positions = np.array(self.config['stop_line_positions'])
            #TODO find the closest visible traffic light (if one exists)
            idxs_waypoint_stop = self.get_closest_waypoint(stop_line_positions)
            
            
            
            # For a valid vehicle pose. Get the closest waypoint of vehicle.         
            if(self.lights) and (self.pose):
                vehicle_pose = np.array([[ self.pose.pose.position.x,self.pose.pose.position.y]])
                idx_waypoint_car = self.get_closest_waypoint(vehicle_pose)        
                
            
                idx_traffic_light = np.argmax((idxs_waypoint_stop-idx_waypoint_car)>0)
                idx_waypoint_traffic_light = idxs_waypoint_stop[idx_traffic_light]
                
                # Get closest traffic light. 
                tmpLight = self.lights[idx_traffic_light]
                
                # Get vehicle pose (position and orientation)
                poseVehicle = self.pose.pose
                
                # Get pose of the waypoint closest to the traffic light. 
                poseLight = self.waypointlist[idx_waypoint_traffic_light,:]
                orientationLight = poseLight[2:]
                positionLight = poseLight[:2]
                
                # Difference between vehicle position and traffic light position. 
                diff_position = positionLight-np.array([ poseVehicle.position.x,poseVehicle.position.y])
                
                
                # Difference in orientation (calculated in quaternions)
                o1 = orientationLight
                o2 = poseVehicle.orientation
                # Angle difference between light-orientation and vehicle-orientation
                q3 = np.quaternion(o1[0],o1[1],o1[2],o1[3]) * np.quaternion(o2.x,o2.y,o2.z,o2.w).inverse()
                diff_angle = euler_from_quaternion(quaternion.as_float_array(q3))
                diff_angle = diff_angle[2]
                
                # Distance between ligt and vehicle
                diff_dist = np.sqrt(np.sum(np.power(diff_position,2)))
                
                print("     dist:", diff_dist ,"diff_position:", diff_position, "angle: ", diff_angle)
                
                
                in_view_dist = 140.0
                in_view_angle = 0.13
                
                
                # Store images with count, label, distance. 
                # e.g. 0010_0_100.jpg is the name of image 10 with label 0 (red light) 100 meter from traffic light
                if self.has_image:
                    
                    # Save factor defines how often an image is stored.
                    # Five-times more images are stored for yellow and green light 
                    save_factor = 10
                    saveImage = False
                    if (tmpLight.state==0): # RED LIGHT --> Slow save factor
                        save_factor = 10
                    elif(tmpLight.state==1):  # Yellow LIGHT --> Faster save factor
                        save_factor = 2
                    elif(tmpLight.state==2):  # Green LIGHT --> Faster save factor
                        save_factor = 2
                    
                    # Defining when a traffic light is in view
                    if (diff_dist < in_view_dist ) and (np.abs(diff_angle)<in_view_angle):          
                        saveImage = True
                        strState = str(tmpLight.state)
                        print("TrafficLight in view: ")
                    
                    # Defining when a traffic light is out of view
                    if (diff_dist > (in_view_dist+100) ) or (np.abs(diff_angle)>(in_view_angle+0.05)):
                        saveImage = True
                        save_factor = 10
                        strState = str(4)
                        print("TrafficLight definitly out off view: ")
                        
                    # Stores images to folder defined in (self.dirData)
                    if saveImage and (np.mod(self.image_count,save_factor)==0):
                        #tmpName = str(np.round(time.time(),decimals=2)).replace(".","_")
                        tmpName = str(self.image_count).zfill(4)
                        dirOut = os.path.join(self.dirData,tmpName+"_"+strState+"_"+ str(int(diff_dist)) + '.jpg')
                        cv2.imwrite(dirOut,self.camera_image)
                        print("SaveImage")
                    
        if light:
            state = self.get_light_state(light)
            return light_wp, state
        
        self.waypoints = None
        
        return idx_waypoint_traffic_light, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
