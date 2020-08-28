from __future__ import print_function

import tf
import copy
import math
import rospy
from std_msgs.msg import Empty, String
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, PointStamped
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from basic_navigation.msg import BasicNavigationFeedback as Feedback

from utils import Utils
from laser_utils import LaserUtils

class BasicNavigation(object):

    """Navigation to move in a straight line towards a goal"""

    def __init__(self):
        # Class variables
        self.plan = None
        self.reached_goal_once = False
        self.moving_backward = False
        self.global_frame = rospy.get_param('~global_frame', 'map')
        self.robot_frame = rospy.get_param('~robot_frame', 'load/base_link')
        self.retry_attempts = 0
        self.current_vel = 0.0
        self.current_mode = None

        self._tf_listener = tf.TransformListener()
        self.laser_utils = LaserUtils(debug=False, only_use_half=True)

        self.default_param_dict_name = rospy.get_param('~default_param_dict_name', 'strict')
        param_dict = rospy.get_param('~' + self.default_param_dict_name)
        self.update_params(param_dict, self.default_param_dict_name)


        # Publishers
        self._cmd_vel_pub = rospy.Publisher('~cmd_vel', Twist, queue_size=1)
        self._path_pub = rospy.Publisher('~path', Path, queue_size=1)
        self._traj_pub = rospy.Publisher('~trajectory', Path, queue_size=1)
        self._feedback_pub = rospy.Publisher('~feedback', Feedback, queue_size=1)

        # Subscribers
        goal_sub = rospy.Subscriber('~goal', PoseStamped, self.goal_cb)
        path_sub = rospy.Subscriber('~goal_path', Path, self.path_cb)
        cancel_goal_sub = rospy.Subscriber('~cancel', Empty, self.cancel_current_goal_cb)
        odom_sub = rospy.Subscriber('~odom', Odometry, self.odom_cb)
        param_sub = rospy.Subscriber('~switch_mode', String, self.switch_mode_cb)


        rospy.sleep(0.5)
        rospy.loginfo('Initialised')

    def run_once(self):
        """
        Main event loop
        """
        self.laser_utils.pub_debug_footprint()

        if self.plan is None:
            return

        curr_pos = self.get_current_position_from_tf()

        if curr_pos is None:
            rospy.logerr('Current pose is not available')
            self._feedback_pub.publish(Feedback(status=Feedback.FAILURE_NO_CURRENT_POSE))
            return

        curr_goal = Utils.get_x_y_theta_from_pose(self.plan[0].pose)
        dist = Utils.get_distance_between_points(curr_pos[:2], curr_goal[:2])
        if len(self.plan) == 1 and (dist < self.goal_dist_tolerance or self.reached_goal_once) :
            self.reached_goal_once = True
            angular_dist = Utils.get_shortest_angle(curr_goal[2], curr_pos[2])
            if abs(angular_dist) < self.goal_theta_tolerance:
                rospy.loginfo('REACHED GOAL')
                self._feedback_pub.publish(Feedback(status=Feedback.SUCCESS))
                self._reset_state()
                return
            else:
                self._rotate_in_place(theta_error=angular_dist)
                return
        if dist < self.waypoint_goal_tolerance and len(self.plan) > 1:
            rospy.logdebug('Reached waypoint')
            reached_wp = self.plan.pop(0)
            self._feedback_pub.publish(Feedback(status=Feedback.REACHED_WP,
                                                reached_wp=reached_wp.pose,
                                                remaining_path_length=len(self.plan)))

        heading = math.atan2(curr_goal[1]-curr_pos[1], curr_goal[0]-curr_pos[0])
        heading_diff = Utils.get_shortest_angle(heading, curr_pos[2])
        if self.allow_backward_motion:
            heading_diff_backward = Utils.get_shortest_angle(
                    heading, Utils.get_reverse_angle(curr_pos[2]))
            if abs(heading_diff) > abs(heading_diff_backward):
                self.moving_backward = True
                heading_diff = heading_diff_backward
            else:
                self.moving_backward = False
        if abs(heading_diff) > self.heading_tolerance:
            self._rotate_in_place(theta_error=heading_diff)
        else:
            final_goal = Utils.get_x_y_theta_from_pose(self.plan[-1].pose)
            total_dist = Utils.get_distance_between_points(curr_pos[:2], final_goal[:2])
            self._move_forward(pos_error=total_dist, theta_error=heading_diff)

    def _rotate_in_place(self, theta_error=1.0):
        theta_vel_raw = theta_error * self.p_theta_in_place
        theta_vel = Utils.clip(theta_vel_raw, self.max_theta_vel, self.min_theta_vel)

        num_of_points = 10
        future_poses = Utils.get_future_poses(0.0, theta_vel, num_of_points,
                                              self.future_pos_lookahead_time)
        collision_index = self.laser_utils.get_collision_index(future_poses)
        if collision_index == 0:
            rospy.logerr('Obstacle while rotating. Current plan failed.')
            self._feedback_pub.publish(Feedback(status=Feedback.FAILURE_OBSTACLES))
            self._reset_state()
            return

        self._cmd_vel_pub.publish(Utils.get_twist(x=0.0, y=0.0, theta=theta_vel))

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
        x_vel = min(desired_x_vel, abs(self.current_vel) + self.max_linear_acc)

        if self.moving_backward:
            x_vel *= -1
        self._cmd_vel_pub.publish(Utils.get_twist(x=x_vel, y=0.0, theta=theta_vel))
        self._traj_pub.publish(Utils.get_path_msg_from_poses(future_poses_with_safety, self.robot_frame))

    def goal_cb(self, msg):
        if self.plan is not None:
            rospy.logwarn('Preempting previous goal. User requested another goal')
            self._feedback_pub.publish(Feedback(status=Feedback.CANCELLED))
        self._reset_state()
        goal = Utils.get_x_y_theta_from_pose(msg.pose)
        rospy.loginfo('Received new goal')
        rospy.loginfo(goal)
        self._get_straight_line_plan(goal)
        param_dict = rospy.get_param('~strict')
        self.update_params(param_dict, 'strict')

    def path_cb(self, msg):
        if self.global_frame != msg.header.frame_id:
            rospy.logwarn('Goal path has "' + msg.header.frame_id + '" frame. '\
                          + 'Expecting "' + self.global_frame + '". Ignoring path.')
            self._feedback_pub.publish(Feedback(status=Feedback.FAILURE_INVALID_PLAN))
            return

        path = msg.poses
        if len(path) == 0:
            rospy.logwarn('Received empty goal path. Ignoring')
            self._feedback_pub.publish(Feedback(status=Feedback.FAILURE_EMPTY_PLAN))
            return

        curr_pos = self.get_current_position_from_tf()
        if curr_pos is None:
            rospy.logerr('Could not get current position. Ignoring path.')
            return 

        first_pose = Utils.get_x_y_theta_from_pose(path[0].pose)
        if self.plan is None:
            dist = Utils.get_distance_between_points(first_pose[:2], curr_pos[:2])
            if dist > self.goal_path_start_point_tolerance:
                rospy.logwarn('Goal path first point is too far from robot. Ignoring path.')
                self._feedback_pub.publish(Feedback(status=Feedback.FAILURE_INVALID_PLAN))
            else:
                rospy.loginfo('Initialising/Replacing current plan')
                self.plan = path
                self._path_pub.publish(msg)
        else:
            curr_plan_last_pose = Utils.get_x_y_theta_from_pose(self.plan[-1].pose)
            plan_append_dist = Utils.get_distance_between_points(first_pose[:2], curr_plan_last_pose[:2])
            if plan_append_dist > self.goal_path_start_point_tolerance:
                rospy.logwarn('Goal path first point is too far from current plan end point. Ignoring.')
                self._feedback_pub.publish(Feedback(status=Feedback.FAILURE_INVALID_PLAN))
            else:
                rospy.loginfo('Appending to current plan')
                self.plan.extend(path)
                self._path_pub.publish(Path(header=msg.header, poses=self.plan))

    def odom_cb(self, msg):
        self.current_vel = msg.twist.twist.linear.x

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

    def _get_straight_line_plan(self, goal):
        """
        Generate a straight line path to reach goal
        """
        rospy.loginfo('Planning straight line motion')
        curr_pos = self.get_current_position_from_tf()
        if curr_pos is None:
            rospy.logerr('Could not get current pose. Ignoring goal')
            return
        path_msg = Path()
        path_msg.header.frame_id = self.global_frame
        path_msg.header.stamp = rospy.Time.now()
        start_pose = Utils.get_pose_stamped_from_frame_x_y_theta(self.global_frame, *curr_pos)
        goal_pose = Utils.get_pose_stamped_from_frame_x_y_theta(self.global_frame, *goal)
        path_msg.poses = [start_pose, goal_pose]
        self.plan = [goal_pose]
        self._path_pub.publish(path_msg)

    def _reset_state(self):
        self.publish_zero_vel()
        self.plan = None
        self.reached_goal_once = False
        self.moving_backward = False
        self.retry_attempts = 0
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
        control_rate = rospy.get_param('~control_rate', 5.0)
        self.max_linear_acc = max_linear_acc_per_second/control_rate
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
