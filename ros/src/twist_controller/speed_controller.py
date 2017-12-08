import rospy
import math
from lowpass import LowPassFilter

GAS_DENSITY = 2.858

class SpeedController(object):
    def __init__(self, wheel_radius, vehicle_mass, fuel_capacity, accel_limit, decel_limit, brake_deadband, carla_low_speed_test):
        self.wheel_radius = wheel_radius
        self.vehicle_mass = vehicle_mass
        self.accel_limit = accel_limit
        self.decel_limit = decel_limit
        self.brake_deadband = brake_deadband

        self.max_torque = vehicle_mass * accel_limit * wheel_radius

        self.filter = LowPassFilter(alpha=0.33)
        self.carla_low_speed_test = carla_low_speed_test

    def get_throttle_brake(self, proposed_linear_vel, current_linear_vel, time_duration):
        if time_duration > 0:
            accel = (proposed_linear_vel - current_linear_vel) / time_duration
            #rospy.logwarn("proposed_vel=%f current_vel=%f carla=%f", proposed_linear_vel, current_linear_vel, self.carla_low_speed_test)

            accel = min(accel, self.accel_limit)
            accel = max(accel, self.decel_limit)

            torque = self.vehicle_mass * accel * self.wheel_radius
            torque = self.filter.filt(torque)

            throttle = torque / self.max_torque

            brake = 0.
            # deceleration => brake cmd
            if throttle < -self.brake_deadband:
                brake = math.fabs(torque)
                if self.carla_low_speed_test: # DBW_TEST
                    # cf https://github.com/udacity/sdc-issue-reports/issues/1204
                    brake /= self.vehicle_mass

            throttle = max(throttle, 0.)
            if self.carla_low_speed_test: # DBW_TEST
                throttle = min(throttle, 0.025)
            #rospy.logwarn("DBWNode: throttle=%f brake=%f", throttle, brake)
            return throttle, brake
        else:
            return 0.0, 0.0
