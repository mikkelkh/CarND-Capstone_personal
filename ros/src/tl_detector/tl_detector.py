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
import cv2
import yaml

import math

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

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.stop_line_positions = self.config['stop_line_positions']
        self.stop_line_waypoints = []

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.ego_x = None
        self.ego_y = None
        self.closest_wp_index = None

        self.num_waypoints = 0
        self.light_wp = -1

        # TEMPORARY just for testing purposes in simulation without real TL detector
        #sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_simu_cb)

        rospy.spin()

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


    def pose_cb(self, msg):
        self.pose = msg
        self.ego_x = msg.pose.position.x
        self.ego_y = msg.pose.position.y

    # this callback is called just once at start
    def waypoints_cb(self, msg):
        if self.num_waypoints == 0:
            self.waypoints = msg.waypoints
            self.num_waypoints = len(self.waypoints)
            for i in range(len(self.stop_line_positions)):
                x = self.stop_line_positions[i][0]
                y = self.stop_line_positions[i][1]
                stop_line_waypoint = self.get_closest_wp_index(x, y, 0, self.num_waypoints)
                self.stop_line_waypoints.append(stop_line_waypoint)
                print(stop_line_waypoint)
            rospy.logwarn("TL_DETECTOR: num_waypoints=%d", self.num_waypoints) # 10902

    def traffic_cb(self, msg):
        self.lights = msg.lights

    # -----------------------------------------------------------------------------------

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
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
        #TODO implement
        return 0

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

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)

        #TODO find the closest visible traffic light (if one exists)

        if light:
            state = self.get_light_state(light)
            return light_wp, state

        # WTF !!!
        #self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
