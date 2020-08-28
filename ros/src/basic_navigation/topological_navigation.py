from __future__ import print_function

import tf
import copy
import time
import math
import rospy

from std_msgs.msg import String, Empty, Bool
from geometry_msgs.msg import PoseStamped, PointStamped, Point
from nav_msgs.msg import Path
from basic_navigation.msg import BasicNavigationFeedback as Feedback

from utils import Utils
from topological_planner import TopologicalPlanner
from geometric_planner import GeometricPlanner
from recovery_manager import RecoveryManager

class TopologicalNavigation(object):

    """Navigation using topology nodes map"""

    def __init__(self):
        # ROS params
        self.global_frame = rospy.get_param('~global_frame', 'map')
        self.robot_frame = rospy.get_param('~robot_frame', 'load/base_link')
        self.geometric_wp_goal_tolerance = rospy.get_param('~geometric_wp_goal_tolerance', 0.5)
        self.topological_wp_goal_tolerance = rospy.get_param('~topological_wp_goal_tolerance', 1.0)
        self.plan_next_geometric_path_tolerance = rospy.get_param('~plan_next_geometric_path_tolerance', 2)
        self.dist_between_wp = rospy.get_param('~dist_between_wp', 1.0)
        network_file = rospy.get_param('~network_file', None)

        # class variables
        self.tf_listener = tf.TransformListener()
        self.topological_planner = TopologicalPlanner(network_file)
        self.geometric_planner = GeometricPlanner(tf_listener=self.tf_listener, debug=True)
        self.recovery_manager = RecoveryManager()
        self.topological_path = None
        self.geometric_path = None
        self.plan_next_geometric_path = False
        self.replan_current_geometric_path = False
        self.feedback_msg = None
        self.is_cart_attached = False

        # subscribers
        goal_sub = rospy.Subscriber('~goal', PoseStamped, self.goal_cb)
        cancel_goal_sub = rospy.Subscriber('~cancel', Empty, self.cancel_current_goal)
        feedback_sub = rospy.Subscriber('~bn_feedback', Feedback, self.feedback_cb)
        cart_attached_sub = rospy.Subscriber('~cart_attached', Bool, self.cart_attached_cb)

        # publishers
        self._path_pub = rospy.Publisher('~topological_path', Path, queue_size=1)
        self._cancel_bn_pub = rospy.Publisher('~cancel_bn', Empty, queue_size=1)
        self._bn_path_pub = rospy.Publisher('~bn_goal_path', Path, queue_size=1)
        self._bn_mode_pub = rospy.Publisher('~bn_mode', String, queue_size=1)
        self._result_pub = rospy.Publisher('~result', Bool, queue_size=1)

        rospy.loginfo('Initialised')

    def run_once(self):
        """
        Main control loop
        """
        if self.topological_path is None:
            return

        if self.feedback_msg is not None:
            self.process_feedback_msg()
            self.feedback_msg = None
            return

        if self.geometric_path is None:
            self.geometric_path = self._get_geometric_path(goal_pose=self.topological_path[0])
            return

        if self.plan_next_geometric_path:
            print('\n\nReached a topological WP\n\n')
            self.plan_next_geometric_path = False
            curr_goal = self.topological_path.pop(0)
            if len(self.topological_path) == 0:
                return
            geometric_path = self._get_geometric_path(start_pose=curr_goal,
                                                      goal_pose=self.topological_path[0])
            if geometric_path is not None:
                if self.geometric_path is not None:
                    self.geometric_path.extend(geometric_path)
                else:
                    self.geometric_path = geometric_path
            else:
                self.geometric_path = None
            return
        
        self.check_future_wp_for_safety()

    def check_future_wp_for_safety(self):
        geometric_x_y_theta_path = [Utils.get_x_y_theta_from_pose(pose.pose) for pose in self.geometric_path]
        safe, collision_index = self.geometric_planner.is_path_safe(geometric_x_y_theta_path)
        if not safe and 0 <= collision_index < 3:
            rospy.logwarn('Path not safe.')
            self.recovery_manager.recover('obstacle', global_navigation_obj=self)
        elif collision_index == -1:
            rospy.logdebug('Path safe')
            self.choose_bn_mode()
            self.recovery_manager.reset(self)

    def process_feedback_msg(self):
        if self.feedback_msg.status == Feedback.SUCCESS:
            if len(self.topological_path) == 0:
                print('\n\nREACHED GOAL\n\n')
                self._result_pub.publish(Bool(data=True))
                self.plan_next_geometric_path = False
                self.topological_path = None
                self._reset_state()
            else:
                self.plan_next_geometric_path = True
        elif self.feedback_msg.status == Feedback.REACHED_WP:
            wp = Utils.get_x_y_theta_from_pose(self.feedback_msg.reached_wp)
            if self.geometric_path is None:
                return
            geometric_path_wp = Utils.get_x_y_theta_from_pose(self.geometric_path[0].pose)
            dist = Utils.get_distance_between_points(wp[:2], geometric_path_wp)
            if dist < self.geometric_wp_goal_tolerance and self.feedback_msg.remaining_path_length == len(self.geometric_path)-1: # verify
                rospy.logdebug('In sync with Basic navigation')
                self.geometric_path.pop(0)
            else:
                rospy.logwarn('Lost sync with Basic navigation.')
                self.recovery_manager.recover('sync', global_navigation_obj=self, feedback_msg=self.feedback_msg)
            if self.feedback_msg.remaining_path_length < self.plan_next_geometric_path_tolerance:
                if len(self.topological_path) > 0:
                    self.plan_next_geometric_path = True
                if len(self.topological_path) < 2:
                    self.choose_bn_mode(dist=self.feedback_msg.remaining_path_length * self.dist_between_wp)
        elif self.feedback_msg.status == Feedback.FAILURE_OBSTACLES:
            rospy.logwarn('Basic navigation failed due to obstacles.')
            self.recovery_manager.recover('obstacle', global_navigation_obj=self)
        elif self.feedback_msg.status == Feedback.FAILURE_EMPTY_PLAN or self.feedback_msg.status == Feedback.FAILURE_INVALID_PLAN:
            rospy.logwarn('Basic navigation received empty or invalid plan. Replanning...')
            self._cancel_bn_pub.publish(Empty())
            self.geometric_path = None
            self.choose_bn_mode()
        elif self.feedback_msg.status == Feedback.FAILURE_NO_CURRENT_POSE:
            rospy.logerr('Basic navigation could not find current pose')
            self.recovery_manager._wait_recovery()

    def feedback_cb(self, msg):
        if self.topological_path is None:
            return
        self.feedback_msg = msg

    def goal_cb(self, msg):
        if self.topological_path is not None:
            rospy.logwarn('Preempting ongoing path. User requested another goal')
        self._reset_state()
        goal = Utils.get_x_y_theta_from_pose(msg.pose)
        rospy.loginfo('Received new goal')
        rospy.loginfo(goal)
        self._get_topological_path(goal)
        if self.topological_path is None:
            return
        rospy.loginfo('Length of topological plan: ' + str(len(self.topological_path)))
        self.choose_bn_mode()

    def cart_attached_cb(self, msg):
        self.is_cart_attached = msg.data
        print('msg', msg)
        param_name = 'cart_attached_footprint' if self.is_cart_attached else 'footprint'
        self.geometric_planner.laser_utils.set_footprint(rospy.get_param('~' + param_name))

    def cancel_current_goal(self, msg):
        """Cancel current goal by sending a cancel signal to basic navigation

        :msg: std_msgs.Empty
        :returns: None

        """
        rospy.logwarn('Preempting')
        self._reset_state()

    def _reset_state(self):
        if self.topological_path is not None:
            self._cancel_bn_pub.publish(Empty())
        self.recovery_manager.reset(self)
        self.topological_path = None
        self.geometric_path = None
        self.plan_next_geometric_path = False
        rospy.sleep(0.5)

    def choose_bn_mode(self, dist=0.0):
        if dist == 0.0 and self.topological_path is not None and len(self.topological_path) > 0:
            curr_pos = self.get_current_position_from_tf()
        
            path = copy.deepcopy(self.topological_path)
            path.insert(0, curr_pos)
            travel_dist = Utils.get_path_length(path)
            rospy.logdebug('Distance of topological plan: ' + str(travel_dist))
        else:
            travel_dist = dist
        if self.is_cart_attached:
            param_name = 'cart'
        else:
            param_name = 'strict' if travel_dist < 3.0 else 'long_dist'
        self._bn_mode_pub.publish(String(data=param_name))
        
    def get_current_position_from_tf(self):
        try:
            trans, rot = self.tf_listener.lookupTransform(self.global_frame,
                                                          self.robot_frame,
                                                          rospy.Time(0))
            _, _, yaw = tf.transformations.euler_from_quaternion(rot)
            curr_pos = (trans[0], trans[1], yaw)
        except Exception as e:
            rospy.logerr(str(e))
            curr_pos = None
        return curr_pos

    def _get_topological_path(self, goal):
        """
        Call topological planner to get point plan from current position to goal pos
        Generates orientation for point plan to create Path msg 
        Publishes Path msg for visualisation

        :goal: tuple of 3 float
        :returns: None

        """
        curr_pos = self.get_current_position_from_tf()
        if curr_pos is None:
            rospy.logerr('Cannot get current position of the robot')
            return None
        rospy.loginfo('Current pos: ' + str(curr_pos))
    
        plan = self.topological_planner.plan(curr_pos[:2], goal[:2])
        if plan is None or len(plan) == 0:
            rospy.logerr('Could not plan topological plan.')
            self._reset_state()
            return

        path_msg = Path()
        path_msg.header.frame_id = self.global_frame
        path_msg.header.stamp = rospy.Time.now()

        theta = 0.0
        self.topological_path = []
        for i in range(len(plan)):
            if i < len(plan)-1:
                theta = math.atan2(plan[i+1].y - plan[i].y, plan[i+1].x - plan[i].x)
            pose = [plan[i].x, plan[i].y, theta]
            pose_stamped = Utils.get_pose_stamped_from_frame_x_y_theta(self.global_frame, *pose)
            pose.append(plan[i].area_type)
            self.topological_path.append(pose)
            path_msg.poses.append(pose_stamped)

        # last rotate in place
        pose = [plan[-1].x, plan[-1].y, goal[2]]
        pose_stamped = Utils.get_pose_stamped_from_frame_x_y_theta(self.global_frame, *pose)
        self.topological_path.append(pose)
        path_msg.poses.append(pose_stamped)

        self._path_pub.publish(path_msg)
        self.topological_path.pop(0) # remove current position
        rospy.loginfo('Planned topological path successfully')

    def _get_geometric_path(self, goal_pose, start_pose=None):
        """
        Call geometric planner to get plan from current position to goal pos
        Publishes Path msg for visualisation

        :goal_pose: tuple of 3 float
        :start_pose: tuple of 3 float
        :returns: None

        """
        if start_pose is None:
            curr_pos = self.get_current_position_from_tf()
            if curr_pos is None:
                rospy.logerr('Cannot get current position of the robot')
                return None
            rospy.loginfo('Current pos: ' + str(curr_pos))
            start_pose = curr_pos
    
        try_spline_first = len(start_pose) > 3 and len(goal_pose) > 3 and start_pose[3] == 'junction' and goal_pose[3] == 'junction'
        plan = self.geometric_planner.plan(start_pose[:3], goal_pose[:3], try_spline_first=try_spline_first)
        if plan is None or len(plan) == 0:
            rospy.logerr('Could not plan geometric plan.')
            self.recovery_manager.recover('plan', global_navigation_obj=self)
            return None

        if len(plan) > 1:
            plan.pop(0)
        path_msg = Utils.get_path_msg_from_poses(plan, self.global_frame)
        geometric_path = path_msg.poses
        self._bn_path_pub.publish(path_msg)
        rospy.loginfo('Planned path successfully')
        return geometric_path
