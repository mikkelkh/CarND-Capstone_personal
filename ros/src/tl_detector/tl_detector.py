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
import yaml
import numpy as np
import os

import math

import math

STATE_COUNT_THRESHOLD = 3

SAVE_IMAGE = False
TRAINING = False

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

        config_string = rospy.get_param("/traffic_light_config")
        
        # Folder of where images are stored. 
        self.dirData = rospy.core.rospkg.environment.get_test_results_dir()
        self.config = yaml.load(config_string)

        self.stop_line_positions = self.config['stop_line_positions']
        self.stop_line_waypoints = []

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.listener = tf.TransformListener()
        self.idx = 0
        self.waypointlist = np.array([])
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.ego_x = None
        self.ego_y = None
        self.closest_wp_index = None

        self.num_waypoints = 0
        self.light_wp = -1
        self.state_red_count = -1

        # Peter and Mikkel
        self.tree = []
        self.image_count = 0
        self.waypoints_prepared = False
        self.idx_traffic_light_found = False
        self.update_waypoint = False
        self.last_pose = None

        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
        
        
        self.light_classifier = TLClassifier()

        # Philippe
#        sub6 = rospy.Subscriber('/image_color', Image, self.image_callback)
#        self.light_classifier = TLClassifier()
#        self.loop()

        rospy.spin()

    # -----------------------------------------------------------------------------------

    def loop(self):
        # every 250 ms is OK on GTX 1080 TI: GPU decoding is arround 120 ms on GTX 1080 TI
        # every 333ms ms, so we have some margin on lower end GPUs: eg on GTX 980 TI decoding is around 250 ms
        # TESTED with GTX 1080 TI => Sould be OK with GTX TITAN X on Carla Self-Driving Car
        rate = rospy.Rate(3)
        while not rospy.is_shutdown():
            if self.camera_image is not None:

                # search wp closest to our car
                if self.closest_wp_index is None:
                    wp_min = 0
                    wp_max = self.num_waypoints - 1
                else:
                    wp_min = self.closest_wp_index - 200
                    wp_max = self.closest_wp_index + 200

                self.closest_wp_index = self.get_closest_wp_index(self.ego_x, self.ego_y, wp_min, wp_max)

                closest_light_red_index = -1
                closest_light_red_dist = 1e10
                for i in range(len(self.lights)):
                    light = self.lights[i]
                    dist = self.stop_line_waypoints[i] - self.closest_wp_index
                    if dist >= 0 and dist < 150 and dist  < closest_light_red_dist:
                        closest_light_red_dist = dist
                        closest_light_red_index = i

                cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
                #Get classification
                state = self.light_classifier.get_classification(cv_image)

                if state == TrafficLight.RED:
                    print('RED')
                    self.light_wp = self.stop_line_waypoints[closest_light_red_index]
                    self.state_red_count = STATE_COUNT_THRESHOLD
                else:
                    print('NOT RED')
                    self.state_red_count -= 1

                if self.state_red_count > 0:
                    print("traffic_waypoint=" + str(self.light_wp))
                    self.upcoming_red_light_pub.publish(Int32(self.light_wp))
                else:
                    self.upcoming_red_light_pub.publish(Int32(-1))

            rate.sleep()


    def image_callback(self, msg):
        if self.num_waypoints > 0 and self.ego_x is not None:
            self.camera_image = msg
            #if (closest_light_red_index >= 0):
            #    # RED or YELLOW light detected
            #    self.light_wp = self.stop_line_waypoints[closest_light_red_index]
            #    self.state_red_count = STATE_COUNT_THRESHOLD
            #else:
            #    self.state_red_count -= 1

    def get_closest_wp_index(self, x, y, wp1, wp2): 
        closest_dist = 1e10
        closest_wp_index = -1
        for i in range(wp1, wp2):
            wp = self.waypoints[i % self.num_waypoints]
            wp_x = wp.pose.pose.position.x
            wp_y = wp.pose.pose.position.y
            dist = math.sqrt( (x - wp_x)**2 + (y - wp_y)**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_wp_index = i
        return closest_wp_index

    # -----------------------------------------------------------------------------------

    def image_simu_cb(self, msg):
        if self.num_waypoints > 0 and self.ego_x is not None:

            # search wp closest to our car
            if self.closest_wp_index is None:
                wp_min = 0
                wp_max = self.num_waypoints - 1
            else:
                wp_min = self.closest_wp_index - 200
                wp_max = self.closest_wp_index + 200
            self.closest_wp_index = self.get_closest_wp_index(self.ego_x, self.ego_y, wp_min, wp_max)

            # simulate traffic light RED detection
            closest_light_red_index = -1
            closest_light_red_dist = 1e10
            for i in range(len(self.lights)):
                light = self.lights[i]
                #if light.state == TrafficLight.RED or light.state == TrafficLight.YELLOW:
                if light.state == TrafficLight.RED:
                    dist = self.stop_line_waypoints[i] - self.closest_wp_index
                    # something realistic in our Field Of View
                    if dist >= 0 and dist < 150 and dist  < closest_light_red_dist:
                        closest_light_red_dist = dist
                        closest_light_red_index = i

            if (closest_light_red_index >= 0):
                # RED or YELLOW light detected
                self.light_wp = self.stop_line_waypoints[closest_light_red_index]
                self.state_red_count = STATE_COUNT_THRESHOLD
            else:
                self.state_red_count -= 1

            if self.state_red_count > 0:
                print("traffic_waypoint=" + str(self.light_wp))
                self.upcoming_red_light_pub.publish(Int32(self.light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(-1))

    def pose_cb(self, msg):
        self.pose = msg
        
        # Philippe
        self.ego_x = msg.pose.position.x
        self.ego_y = msg.pose.position.y

    # this callback is called just once at start
    def waypoints_cb(self, msg):
        # Peter and Mikkel
        rospy.loginfo("WAYPOINTS_RECEIVED")
        self.update_waypoint = True
    
        # Philippe
        if self.num_waypoints == 0:
            self.waypoints = msg.waypoints
            self.num_waypoints = len(self.waypoints)
            for i in range(len(self.stop_line_positions)):
                x = self.stop_line_positions[i][0]
                y = self.stop_line_positions[i][1]
                stop_line_waypoint = self.get_closest_wp_index(x, y, 0, self.num_waypoints)
                self.stop_line_waypoints.append(stop_line_waypoint)
                print(stop_line_waypoint)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    # -----------------------------------------------------------------------------------

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        if self.num_waypoints > 0 and self.ego_x is not None:
            self.camera_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.has_image = True
            self.image_count = self.image_count+1
            light_wp, state = self.process_traffic_lights()
    
    
            if state == TrafficLight.RED:
                print('RED')
                self.light_wp = self.stop_line_waypoints[light_wp]
                self.state_red_count = STATE_COUNT_THRESHOLD
            else:
                print('NOT RED')
                self.state_red_count -= 1

            if self.state_red_count > 0:
                print("traffic_waypoint=" + str(self.light_wp))
                self.upcoming_red_light_pub.publish(Int32(self.light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(-1))    
    
            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise the previous stable state is
            used.
            '''
#            if self.state != state:
#                self.state_count = 0
#                self.state = state
#            elif self.state_count >= STATE_COUNT_THRESHOLD:
#                self.last_state = self.state
#                light_wp = light_wp if state == TrafficLight.RED else -1
#                self.last_wp = light_wp
#                self.upcoming_red_light_pub.publish(Int32(light_wp))
#            else:
#                self.upcoming_red_light_pub.publish(Int32(self.last_wp))
#            self.state_count += 1


#    def get_closest_waypoint(self, pose):
#        """Identifies the closest path waypoint to the given position
#            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
#        Args:
#            pose (Pose): position to match a waypoint to
#
#        Returns:
#            int: index of the closest waypoint in self.waypoints
#
#        """
#
#        # The closest waypoint is determiend using a KDtree - just to be difficult.
#        useKDTree = True
#        if(self.waypoints_prepared == False) or self.update_waypoint:
#            # Placing waypoints in list
#            self.waypointlist = np.array([[waypoint.pose.pose.position.x,
#                                           waypoint.pose.pose.position.y,
#                                           waypoint.pose.pose.orientation.x,
#                                           waypoint.pose.pose.orientation.y,
#                                           waypoint.pose.pose.orientation.z,
#                                           waypoint.pose.pose.orientation.w] for waypoint in self.waypoints])
#            
#            # Determining the KDTree to faster measure distance to closest point
#            if useKDTree:
#                self.tree = spatial.KDTree(self.waypointlist[:,0:2],1)
#            self.waypoints_prepared = True
#            self.update_waypoint = False
#            
#        
#        # Get closest idx. 
#        if(self.waypoints_prepared):
#            if useKDTree:
#                dist,idxs = self.tree.query(pose)
#                self.idx = idxs
#            else:
#                # Otherwise calculate it
#                dist = np.sum(np.power(pose-self.waypointlist[:,0:2],2),axis=1)
#                self.idx = np.argmin(dist)
#        else:
#            self.idx = 0
#            dist = 0
#            
#        #print("GetClosestWayPoint: ", self.idx,dist)
#        return self.idx

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
        if TRAINING is True:
            return TrafficLight.UNKNOWN
        else:
            return self.light_classifier.get_classification(self.camera_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

#        idx_waypoint_traffic_light = -1
        
        
        # This is only processed if waypoints have been received. 
#        if self.num_waypoints > 0:
            
            # List of positions that correspond to the line to stop in front of for a given intersection
#            stop_line_positions = np.array(self.config['stop_line_positions'])
            #TODO find the closest visible traffic light (if one exists)
#            idxs_waypoint_stop = self.get_closest_waypoint(stop_line_positions)
            
        # For a valid vehicle pose. Get the closest waypoint of vehicle.         
        if (self.lights) and (self.pose):
#                vehicle_pose = np.array([[ self.pose.pose.position.x,self.pose.pose.position.y]])
#                idx_waypoint_car = self.get_closest_waypoint(vehicle_pose)        
#                
#            
#                idx_traffic_light = np.argmax((idxs_waypoint_stop-idx_waypoint_car)>0)
#                idx_waypoint_traffic_light = idxs_waypoint_stop[idx_traffic_light]
#                
#                # Get closest traffic light. 
#                tmpLight = self.lights[idx_traffic_light]
#                
#                # Get pose of the waypoint closest to the traffic light. 
#                poseLight = self.waypointlist[idx_waypoint_traffic_light,:]
#                orientationLight = poseLight[2:]
#                positionLight = poseLight[:2]
#                
#                # Difference between vehicle position and traffic light position. 
#                diff_position = positionLight-np.array(vehicle_pose)
#                
#                # Difference in orientation (calculated in quaternions)
##                o1 = orientationLight
##                o2 = self.pose.pose.orientation
#                
#                # Angle difference between light-orientation and vehicle-orientation
##                q3 = np.quaternion(o1[0],o1[1],o1[2],o1[3]) * np.quaternion(o2.x,o2.y,o2.z,o2.w).inverse()
##                q3 = quaternion_multiply([o1[0],o1[1],o1[2],o1[3]],[o2.x,o2.y,o2.z,o2.w])
##                q3 = tf.transformations.unit_vector(q3)
##                q3 = quaternion_multiply([o1[0],o1[1],o1[2],o1[3]],[o2[0],o2[1],o2[2],o2[3]])
##                diff_angle = euler_from_quaternion(quaternion.as_float_array(q3))[2]
##                diff_angle = euler_from_quaternion(q3)[2]
#                
##                q3_2 = np.quaternion(o1[0],o1[1],o1[2],o1[3]) * np.quaternion(o2.x,o2.y,o2.z,o2.w).inverse()
##                diff_angle_2 = euler_from_quaternion(quaternion.as_float_array(q3_2))[2]
#                
#                # Distance between ligt and vehicle
#                diff_dist = np.sqrt(np.sum(np.power(diff_position,2)))
#                
##                rospy.loginfo("dist: %s, diff_position: %s, angle: %s, angle2=%s", diff_dist, diff_position, diff_angle, diff_angle_2)
#                rospy.loginfo("dist: %s, diff_position: %s", diff_dist, diff_position)
#                
##                in_view_dist = 140.0
#                out_of_view_max = 140.0
#                out_of_view_min = 10.0
##                in_view_angle = 0.13
#                
#                
#                # Store images with count, label, distance. 
#                # e.g. 0010_0_100.jpg is the name of image 10 with label 0 (red light) 100 meter from traffic light
#                if self.has_image:
#                    
#                    # Save factor defines how often an image is stored.
#                    # Five-times more images are stored for yellow and green light 
#                    save_factor = 10
#                    saveImage = False
#                    if (tmpLight.state==0): # RED LIGHT --> Slow save factor
#                        save_factor = 10
#                    elif(tmpLight.state==1):  # Yellow LIGHT --> Faster save factor
#                        save_factor = 2
#                    elif(tmpLight.state==2):  # Green LIGHT --> Faster save factor
#                        save_factor = 2
#                    
#                    # Defining when a traffic light is in view
#                    if ((diff_dist <= out_of_view_max) and (diff_dist >= out_of_view_min)): # and (np.abs(diff_angle)<in_view_angle):          
#                        saveImage = True
#                        strState = str(tmpLight.state)
#                        rospy.loginfo("TrafficLight in view: ")
#                    
#                    # Defining when a traffic light is out of view
#                    if (diff_dist >= (out_of_view_max+100) ): # or (np.abs(diff_angle)>(in_view_angle+0.05)):
#                        saveImage = True
#                        save_factor = 10
#                        strState = str(4)
#                        rospy.loginfo("TrafficLight definitly out off view: ")
#                        
#                    # Stores images to folder defined in (self.dirData)
#                    if saveImage and (np.mod(self.image_count,save_factor)==0):
#                        #tmpName = str(np.round(time.time(),decimals=2)).replace(".","_")
#                        tmpName = str(self.image_count).zfill(4)
#                        dirOut = os.path.join(self.dirData,tmpName+"_"+strState+"_"+ str(int(diff_dist)) + '.jpg')
#                        cv2.imwrite(dirOut,self.camera_image)
#                        rospy.loginfo("SaveImage")
                    
                    
                    
            # search wp closest to our car
            if self.closest_wp_index is None:
                wp_min = 0
                wp_max = self.num_waypoints - 1
            else:
                wp_min = self.closest_wp_index - 200
                wp_max = self.closest_wp_index + 200
            self.closest_wp_index = self.get_closest_wp_index(self.ego_x, self.ego_y, wp_min, wp_max)

            # simulate traffic light RED detection
            closest_light_index = -1
            closest_light_dist = 1e10
            for i in range(len(self.lights)):
#                    light = self.lights[i]
                #if light.state == TrafficLight.RED or light.state == TrafficLight.YELLOW:
#                    if light.state == TrafficLight.RED:
                dist = self.stop_line_waypoints[i] - self.closest_wp_index
                # something realistic in our Field Of View
                if dist >= 0 and dist < 200 and dist  < closest_light_dist:
                    closest_light_dist = dist
                    closest_light_index = i
                    
            light = self.lights[closest_light_index]

            if SAVE_IMAGE and self.has_image:       
                strState = str(light.state)

                # Is the traffic light visible in the image?
                if (closest_light_index >= 0):
                    light_visible = True
                else:
                    light_visible = False
                    
                if (light_visible==False):
                    save_factor = 30
                    strState = str(4)
                elif (light.state==0): # RED LIGHT --> Slow save factor
                    save_factor = 10
                elif(light.state==1):  # Yellow LIGHT --> Faster save factor
                    save_factor = 1
                elif(light.state==2):  # Green LIGHT --> Faster save factor
                    save_factor = 1
                    
                # Is the car moving? (only save images when it is)
                if self.last_pose is not None:
                    diff_dist = math.sqrt((self.pose.pose.position.x-self.last_pose.pose.position.x)**2+(self.pose.pose.position.y-self.last_pose.pose.position.y)**2+(self.pose.pose.position.z-self.last_pose.pose.position.z)**2)
                else:
                    diff_dist = 0
                    
                if (diff_dist > 0.001) and (np.mod(self.image_count,save_factor)==0):
                    # Stores images to test results folder
                    tmpName = str(self.image_count).zfill(4)
                    dirOut = os.path.join(self.dirData,tmpName+"_"+strState+"_"+ str(int(closest_light_dist)) + '.jpg')
                    cv2.imwrite(dirOut,self.camera_image)
                    rospy.loginfo("SaveImage")
                          
        self.last_pose = self.pose
                    
        light_wp = closest_light_index
#        light = None
        if light:
            state = self.get_light_state(light)
            return light_wp, state

        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
