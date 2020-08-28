from __future__ import print_function

import tf
import copy
import math
import rospy
from std_msgs.msg import Empty, String
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, PointStamped
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped

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
                theta_vel = Utils.clip(theta_vel_raw, self.max_theta_vel, -self.max_theta_vel)
                self.pub_ramped_vel(0.0, 0.0, theta_vel)
                return

        goal = Utils.transform_pose(self._tf_listener, self.goal,
                                    self.global_frame, self.robot_frame)
        goal_force = Utils.get_normalised(goal[:2])
        # goal_force = [Utils.clip(goal[0], 1.0, -1.0), Utils.clip(goal[1], 1.0, -1.0)]

        force_vector = Utils.get_normalised([goal_force[0]+obstacle_force[0],
                                             goal_force[1]+obstacle_force[1]])
        heading = math.atan2(self.goal[1]-curr_pos[1], self.goal[0]-curr_pos[0])
        heading_diff = Utils.get_shortest_angle(heading, curr_pos[2])
        if abs(heading_diff) > self.heading_tolerance:
            theta_vel_raw = heading_diff * self.p_theta_in_place
            theta_vel = Utils.clip(theta_vel_raw, self.max_theta_vel, -self.max_theta_vel)
            self.pub_ramped_vel(0.0, 0.0, theta_vel)
            return

        x_vel = force_vector[0]*max(dist, 0.5)*self.max_linear_vel
        y_vel = force_vector[1]*max(dist, 0.5)*self.max_linear_vel
        theta_vel = math.atan2(force_vector[1], force_vector[0]) * self.p_theta
        self.pub_ramped_vel(x_vel, y_vel, theta_vel)

    def get_obstacle_force(self):
        force_vector = [0.0, 0.0]
        pts_within_neighbourhood = 0
        for p in self.laser_utils.points:
            if Utils.get_distance(*p) < 1.0:
                force_vector[0] -= p[0]
                force_vector[1] -= p[1]
                pts_within_neighbourhood += 1
        if pts_within_neighbourhood == 0:
            return [0.0, 0.0]
        return Utils.get_normalised(force_vector)

    def _move_forward(self, pos_error=1.0, theta_error=1.0):
        theta_vel_raw = theta_error * self.p_theta
        theta_vel = Utils.clip(theta_vel_raw, self.max_theta_vel, self.min_theta_vel)

        future_vel_prop_raw = (pos_error * self.p_linear) / (1.0 + abs(theta_vel) * self.c_theta)
        future_vel_prop = Utils.clip(future_vel_prop_raw, self.max_linear_vel, self.min_linear_vel)

        num_of_points = 10
        future_poses = Utils.get_future_poses(future_vel_prop, theta_vel, num_of_points,
                                              self.future_pos_lookahead_time)
        future_poses_with_safety = [list(pose) for pose in future_poses]
        for pose in future_poses_with_safety:
            if self.moving_backward:
                pose[0] *= -1
                pose[0] -= self.forward_safety_dist
            else:
                pose[0] += self.forward_safety_dist
        self.laser_utils.use_front_half = not self.moving_backward
        collision_index = self.laser_utils.get_collision_index(future_poses_with_safety)

        if collision_index == 0:
            rospy.logerr('Obstacle ahead. Current plan failed.')
            self._feedback_pub.publish(Feedback(status=Feedback.FAILURE_OBSTACLES))
            self._reset_state()
            return

        desired_x_vel = future_vel_prop * (float(collision_index)/num_of_points)
        desired_x_vel = Utils.clip(desired_x_vel, self.max_linear_vel, self.min_linear_vel)

        # ramp up the vel according to max_linear_acc
        x_vel = min(desired_x_vel, abs(self._current_vel[0]) + self.max_linear_acc)

        if self.moving_backward:
            x_vel *= -1
        self._cmd_vel_pub.publish(Utils.get_twist(x=x_vel, y=0.0, theta=theta_vel))
        self._traj_pub.publish(Utils.get_path_msg_from_poses(future_poses_with_safety, self.robot_frame))

    def pub_ramped_vel(self, x_vel=0.0, y_vel=0.0, theta_vel=0.0):
        num_of_points = 10
        x_vel = Utils.clip(x_vel, self.max_linear_vel, -self.max_linear_vel)
        y_vel = Utils.clip(y_vel, self.max_linear_vel, -self.max_linear_vel)
        theta_vel = Utils.clip(theta_vel, self.max_theta_vel, -self.max_theta_vel)
        # future_poses = Utils.get_future_poses(x_vel, theta_vel, num_of_points,
        #                                       self.future_pos_lookahead_time)
        future_poses = Utils.get_future_poses_holonomic(x_vel, y_vel, num_of_points,
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
        # self._current_vel = (msg.linear.x, msg.linear.y, msg.angular.z)

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
        self.forward_safety_dist = param_dict.get('forward_safety_dist', 0.1)
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
