from __future__ import print_function

import tf
import copy
import math
import rospy
from std_msgs.msg import Empty, String
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, PointStamped
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from visualization_msgs.msg import MarkerArray, Marker

from utils import Utils
from laser_utils import LaserUtils

class VectorFieldNavigation(object):

    """Navigation using vector field towards a goal"""

    def __init__(self):
        # Class variables
        self.plan = None
        self.reached_goal_once = False
        self.moving_backward = False
        self.global_frame = rospy.get_param('~global_frame', 'map')
        self.robot_frame = rospy.get_param('~robot_frame', 'load/base_link')
        self._current_vel = (0.0, 0.0, 0.0)
        self.current_mode = None
        self.goal = None

        self._tf_listener = tf.TransformListener()
        self.laser_utils = LaserUtils(debug=False, only_use_half=True)

        self.default_param_dict_name = rospy.get_param('~default_param_dict_name', 'strict')
        param_dict = rospy.get_param('~' + self.default_param_dict_name)
        self.update_params(param_dict, self.default_param_dict_name)

        # Publishers
        self._cmd_vel_pub = rospy.Publisher('~cmd_vel', Twist, queue_size=1)
        self._path_pub = rospy.Publisher('~path', Path, queue_size=1)
        self._traj_pub = rospy.Publisher('~trajectory', Path, queue_size=1)
        self._debug_field_pub = rospy.Publisher('~field', MarkerArray, queue_size=1)

        # Subscribers
        goal_sub = rospy.Subscriber('~goal', PoseStamped, self.goal_cb)
        cancel_goal_sub = rospy.Subscriber('~cancel', Empty, self.cancel_current_goal_cb)
        odom_sub = rospy.Subscriber('~odom', Odometry, self.odom_cb)
        # odom_sub = rospy.Subscriber('~cmd_vel', Twist, self.odom_cb)
        param_sub = rospy.Subscriber('~switch_mode', String, self.switch_mode_cb)

        rospy.sleep(0.5)
        rospy.loginfo('Vector Field Navigation Initialised')

    def run_once(self):
        """
        Main event loop
        """
        self.laser_utils.pub_debug_footprint()

        if self.goal is None:
            return

        curr_pos = self.get_current_position_from_tf()

        if curr_pos is None:
            rospy.logerr('Current pose is not available')
            self.goal = None
            return

        obstacle_force = self.get_obstacle_force()

        dist = Utils.get_distance_between_points(curr_pos[:2], self.goal[:2])
        if dist < self.goal_dist_tolerance or self.reached_goal_once:
            self.reached_goal_once = True
            angular_dist = Utils.get_shortest_angle(self.goal[2], curr_pos[2])
            if abs(angular_dist) < self.goal_theta_tolerance:
                rospy.loginfo('REACHED GOAL')
                self._reset_state()
                return
            else:
                theta_vel_raw = angular_dist * self.p_theta_in_place
                theta_vel = Utils.signed_clip(theta_vel_raw, self.max_theta_vel, -self.max_theta_vel)
                self.pub_ramped_vel(0.0, 0.0, theta_vel)
                return

        goal = Utils.transform_pose(self._tf_listener, self.goal,
                                    self.global_frame, self.robot_frame)
        # goal_force = Utils.get_normalised(goal[:2])
        if obstacle_force == [0.0, 0.0]:
            goal_force_upper_limit = 1.0
        else:
            goal_force_upper_limit = 0.8
        goal_force = [Utils.signed_clip(goal[0], goal_force_upper_limit, 0.1),
                      Utils.clip(goal[1], goal_force_upper_limit, -0.8)]

        force_vector = [goal_force[0]+obstacle_force[0],
                        goal_force[1]+obstacle_force[1]]

        heading = math.atan2(*goal_force[::-1])
        linear_scale = 1.0 - (abs(heading)/math.pi)**2

        x_vel = force_vector[0]*self.max_linear_vel*linear_scale
        y_vel = force_vector[1]*self.max_linear_vel*linear_scale
        theta_vel =  heading * self.p_theta
        self.pub_force_field()
        # print("goal", round(goal_force[0], 3), round(goal_force[1], 3))
        # print("obts", round(obstacle_force[0], 3), round(obstacle_force[1], 3))
        # print("totl", round(force_vector[0], 3), round(force_vector[1], 3))
        # print()
        # x_vel = obstacle_force[0]*self.max_linear_vel
        # y_vel = obstacle_force[1]*self.max_linear_vel
        # theta_vel = 0.0
        self.pub_ramped_vel(x_vel, y_vel, theta_vel)

    def get_obstacle_force(self, x=0.0, y=0.0):
        force_vector = [0.0, 0.0]
        pts_within_neighbourhood = 0
        obstacle_range = self.neighbourhood_dist - self.safety_dist
        for p in self.laser_utils.points:
            pt = [p[0]-x, p[1]-y]
            dist = Utils.get_distance(*pt)
            if dist < self.neighbourhood_dist:
                force_vector[0] -= (pt[0]/dist) *\
                                   ((self.neighbourhood_dist - dist)/(obstacle_range))**2
                force_vector[1] -= (pt[1]/dist) *\
                                   ((self.neighbourhood_dist - dist)/(obstacle_range))**2
                pts_within_neighbourhood += 1
        if pts_within_neighbourhood < self.neighbourhood_pts_threshold:
            return [0.0, 0.0]

        return force_vector

    def pub_ramped_vel(self, x_vel=0.0, y_vel=0.0, theta_vel=0.0):
        num_of_points = 10
        x_vel = Utils.clip(x_vel, self.max_linear_vel, -self.max_linear_vel)
        y_vel = Utils.clip(y_vel, self.max_linear_vel, -self.max_linear_vel)
        theta_vel = Utils.clip(theta_vel, self.max_theta_vel, -self.max_theta_vel)
        future_poses = Utils.get_future_poses(x_vel, y_vel,theta_vel, num_of_points,
                                              self.future_pos_lookahead_time)
        self._traj_pub.publish(Utils.get_path_msg_from_poses(future_poses, self.robot_frame))
        x_vel = Utils.clip(x_vel, self._current_vel[0]+self.max_linear_acc,
                                  self._current_vel[0]-self.max_linear_acc)
        y_vel = Utils.clip(y_vel, self._current_vel[1]+self.max_linear_acc,
                                  self._current_vel[1]-self.max_linear_acc)
        theta_vel = Utils.clip(theta_vel, self._current_vel[2]+self.max_angular_acc,
                                          self._current_vel[2]-self.max_angular_acc)
        self._cmd_vel_pub.publish(Utils.get_twist(x=x_vel, y=y_vel, theta=theta_vel))

    def goal_cb(self, msg):
        self._reset_state()
        if self.goal is not None:
            rospy.logwarn('Preempting previous goal. User requested another goal')
        if msg.header.frame_id != self.global_frame:
            rospy.logwarn('Goal not in correct frame. Expecting in ' + str(self.global_frame))
            return
        self.goal = Utils.get_x_y_theta_from_pose(msg.pose)
        rospy.loginfo('Received new goal')
        rospy.loginfo(self.goal)

    def odom_cb(self, msg):
        self._current_vel = (msg.twist.twist.linear.x,
                             msg.twist.twist.linear.y,
                             msg.twist.twist.angular.z)

    def get_current_position_from_tf(self):
        try:
            trans, rot = self._tf_listener.lookupTransform(self.global_frame,
                                                           self.robot_frame,
                                                           rospy.Time(0))
            _, _, yaw = tf.transformations.euler_from_quaternion(rot)
            curr_pos = (trans[0], trans[1], yaw)
        except Exception as e:
            rospy.logerr(str(e))
            curr_pos = None
        return curr_pos

    def _reset_state(self):
        self.goal = None
        self.reached_goal_once = False
        self.publish_zero_vel()
        self.moving_backward = False
        param_dict = rospy.get_param('~' + self.default_param_dict_name)
        self.update_params(param_dict, self.default_param_dict_name)

    def update_params(self, param_dict, param_name=''):
        rospy.loginfo('Updating params to ' + param_name)
        self.current_mode = param_name
        self.allow_backward_motion = param_dict.get('allow_backward_motion', False)
        footprint = param_dict.get('footprint', [[-0.33, 0.33], [0.33, 0.33], [0.33, -0.33], [-0.33, -0.33]])
        self.laser_utils.set_footprint(footprint)
        footprint_padding = param_dict.get('footprint_padding', 0.1)
        self.laser_utils.set_footprint_padding(footprint_padding)

        # tolerances
        self.heading_tolerance = param_dict.get('heading_tolerance', 0.5)
        self.goal_dist_tolerance = param_dict.get('goal_dist_tolerance', 0.1)
        self.goal_theta_tolerance = param_dict.get('goal_theta_tolerance', 0.1)
        self.waypoint_goal_tolerance = param_dict.get('waypoint_goal_tolerance', 0.3)
        self.safety_dist = param_dict.get('safety_dist', 0.5)
        self.latch_xy_goal = param_dict.get('latch_xy_goal', True)
        self.goal_path_start_point_tolerance = param_dict.get('goal_path_start_point_tolerance', 1.0)

        # controller params
        self.p_theta_in_place = param_dict.get('p_theta_in_place', 5.0)
        self.p_theta = param_dict.get('p_theta', 1.0)
        self.c_theta = param_dict.get('c_theta', 100.0)
        self.p_linear = param_dict.get('p_linear', 1.0)
        self.max_theta_vel = param_dict.get('max_theta_vel', 0.5)
        self.min_theta_vel = param_dict.get('min_theta_vel', 0.005)
        self.max_linear_vel = param_dict.get('max_linear_vel', 0.3)
        self.min_linear_vel = param_dict.get('min_linear_vel', 0.1)
        max_linear_acc_per_second = param_dict.get('max_linear_acc', 0.5)
        max_angular_acc_per_second = param_dict.get('max_angular_acc', 0.5)
        control_rate = rospy.get_param('~control_rate', 5.0)
        self.max_linear_acc = max_linear_acc_per_second/control_rate
        self.max_angular_acc = max_angular_acc_per_second/control_rate
        self.future_pos_lookahead_time = param_dict.get('future_pos_lookahead_time', 3.0)
        self.neighbourhood_dist = param_dict.get('neighbourhood_dist', 1.0)
        self.neighbourhood_pts_threshold = param_dict.get('neighbourhood_pts_threshold', 3)

    def switch_mode_cb(self, msg):
        mode = msg.data
        if mode == self.current_mode:
            return
        if mode in ['long_dist', 'strict', 'cart']:
            param_dict = rospy.get_param('~' + mode)
            self.update_params(param_dict, mode)
        else:
            print('unknown mode')
            param_dict = rospy.get_param('~' + self.default_param_dict_name)
            self.update_params(param_dict, self.default_param_dict_name)

    def cancel_current_goal_cb(self, msg):
        rospy.logwarn('PREEMPTING (cancelled goal)')
        self._feedback_pub.publish(Feedback(status=Feedback.CANCELLED))
        self._reset_state()

    def publish_zero_vel(self):
        self._cmd_vel_pub.publish(Utils.get_twist())

    def pub_force_field(self):
        limit = 1.5
        delta = 0.3
        x, y = -limit, -limit
        force_field = []
        while x <= limit:
            while y <= limit:
                force_vector = self.get_obstacle_force(x, y)
                force_field.append(((x, y), force_vector))
                y += delta
            x += delta
            y = -limit
        self._debug_field_pub.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
        marker_array_msg = VectorFieldNavigation.get_field_msg(force_field, self.robot_frame)
        self._debug_field_pub.publish(marker_array_msg)

    @staticmethod
    def get_field_msg(force_field, frame):
        marker_array_msg = MarkerArray()
        for i, (position, force_vector) in enumerate(force_field):
            force_magnitude = Utils.get_distance(*force_vector)
            arrow_marker = Marker(id=i, ns='field')
            arrow_marker.type = Marker.ARROW
            arrow_marker.scale.x = 0.005 * force_magnitude
            arrow_marker.scale.y = 0.001 * force_magnitude
            arrow_marker.scale.z = 0.001 * force_magnitude
            arrow_marker.color.r = 1.0
            arrow_marker.color.g = 0.0
            arrow_marker.color.b = 0.0
            arrow_marker.color.a = 1.0
            arrow_marker.pose = Utils.get_pose_from_x_y_theta(position[0], position[1],
                                                              math.atan2(-position[1], -position[0]))
            arrow_marker.header.stamp = rospy.Time.now()
            arrow_marker.header.frame_id = frame
            marker_array_msg.markers.append(arrow_marker)

        return marker_array_msg
