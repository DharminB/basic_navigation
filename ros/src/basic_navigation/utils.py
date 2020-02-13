from __future__ import print_function

import tf
import math
import rospy
from geometry_msgs.msg import PoseStamped, Pose, TransformStamped, Quaternion
from geometry_msgs.msg import Twist

class Utils(object):

    """Utility functions used for basic 2d navigation"""

    @staticmethod
    def get_pose_from_x_y_theta(x, y, theta):
        """Return a Pose object from x, y and theta
        :x: float
        :y: float
        :theta: float
        :returns: geometry_msgs.Pose

        """
        pose = Pose()
        pose.position.x = x
        pose.position.y = y

        quat = tf.transformations.quaternion_from_euler(0.0, 0.0, theta)
        pose.orientation = Quaternion(*quat)
        return pose

    @staticmethod
    def get_pose_stamped_from_frame_x_y_theta(frame, x, y, theta):
        """Return a Pose object from x, y and theta
        :x: float
        :y: float
        :theta: float
        :frame: string
        :returns: geometry_msgs.PoseStamped

        """
        pose = PoseStamped()
        pose.pose = Utils.get_pose_from_x_y_theta(x, y, theta)
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = frame
        return pose

    @staticmethod
    def get_x_y_theta_from_pose(pose):
        """Return a tuple(x, y, theta) from Pose objects

        :pose: geometry_msgs/Pose
        :returns: tuple(x, y, theta)

        """
        quat = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        theta = tf.transformations.euler_from_quaternion(quat)[2]
        return (pose.position.x, pose.position.y, theta)

    @staticmethod
    def get_static_transform_from_x_y_theta(x, y, theta, frame_id="odom"):
        """Create a TransformedStamped message from x y and theta to 0, 0 and 0

        :x: float
        :y: float
        :theta: float
        :returns: geometry_msgs.TransformStamped
        """
        transform = TransformStamped()

        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = frame_id
        transform.child_frame_id = "start_pose"

        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = 0.0

        quat = tf.transformations.quaternion_from_euler(0.0, 0.0, theta)
        transform.transform.rotation = Quaternion(*quat)
        return transform

    @staticmethod
    def get_shortest_angle(angle1, angle2):
        """Compute the angular distance between two angles (in radians)

        :angle1: float
        :angle2: float
        :returns: float
        """
        return math.atan2(math.sin(angle1 - angle2), math.cos(angle1 - angle2))

    @staticmethod
    def get_reverse_angle(angle):
        """Compute the angle facing opposite of given angle and ensures that the 
        returned angle is between pi and -pi
        ASSUMPTION: angle is always between pi and -pi

        :angle: float
        :returns: float
        """
        reverse_angle = angle - math.pi
        if reverse_angle < -math.pi:
            reverse_angle = reverse_angle + (2 * math.pi)
        return reverse_angle

    @staticmethod
    def get_distance(delta_x, delta_y):
        """Compute cartesian distance given individual distance in x and y axis

        :delta_x: float
        :delta_y: float
        :returns: float

        """
        return (delta_x**2 + delta_y**2)**0.5

    @staticmethod
    def get_distance_between_points(p1, p2):
        """Compute cartesian distance given two points

        :p1: tuple(float, float)
        :p2: tuple(float, float)
        :returns: float

        """
        return Utils.get_distance(p1[0]-p2[0], p1[1]-p2[1])

    @staticmethod
    def clip(value, max_allowed=1.0, min_allowed=0.1):
        """Clip the provided value to be between the given range

        :value: float
        :max_allowed: float
        :min_allowed: float
        :returns: float

        """
        sign = 1.0 if value > 0.0 else -1.0
        return sign * min(max_allowed, max(min_allowed, abs(value)))

    @staticmethod
    def get_twist(x=0.0, y=0.0, theta=0.0):
        """Return twist ros message object.

        :x: float
        :y: float
        :theta: float
        :returns: geometry_msgs.msg.Twist

        """
        msg = Twist()
        msg.linear.x = x
        msg.linear.y = y
        msg.angular.z = theta
        return msg
