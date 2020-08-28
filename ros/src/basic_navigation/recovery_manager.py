from __future__ import print_function

import math
import rospy
from nav_msgs.msg import Path
from std_msgs.msg import String, Empty
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from utils import Utils

class RecoveryManager(object):

    """Monitor and execute recovery behaviours"""

    def __init__(self):
        # ros params
        self.recovery_wait_duration = rospy.get_param('~recovery_wait_duration', 1.0)
        self.recovery_motion_duration = rospy.get_param('~recovery_motion_duration', 2.0)
        self.recovery_vel = rospy.get_param('~recovery_vel', -0.1)
        self.default_footprint_padding = rospy.get_param('~footprint_padding', 0.1)

        # class variables
        self.recovery_level_dict = {'obstacle': 1, 'plan': 1, 'sync': 1}
        self._level_to_func = {1:'_wait_recovery', 2:'_replan_recovery',
                              3:'_reconfigure_recovery', 4:'_move_away_recovery',
                              5:'_modify_goal_recovery', 6: '_failure'}
        self._type_to_func = {'obstacle': 'recover_from_obstacle_failure',
                              'plan': 'recover_from_plan_failure',
                              'sync': 'recover_from_sync_failure'}
        self.move_away_recovery_counter = 0

        # publisher
        self._cmd_vel_pub = rospy.Publisher('~cmd_vel', Twist, queue_size=1)

    def reset(self, global_navigation_obj):
        if self.recovery_level_dict != {'obstacle': 1, 'plan': 1, 'sync': 1}:
            self.recovery_level_dict = {'obstacle': 1, 'plan': 1, 'sync': 1}
            global_navigation_obj.geometric_planner.laser_utils.set_footprint_padding(self.default_footprint_padding)
            self.move_away_recovery_counter = 0
            global_navigation_obj.choose_bn_mode()
        
    def recover(self, failure_type, **kwargs):
        rospy.loginfo('RECOVERING')
        func_name = self._type_to_func.get(failure_type, None)
        if func_name is not None:
            getattr(self, func_name)(**kwargs)
        rospy.loginfo('Recovery finished')

    def recover_from_obstacle_failure(self, **kwargs):
        rospy.loginfo('Obstacle recovery')
        current_level = self.recovery_level_dict.get('obstacle')
        func_name = self._level_to_func.get(current_level, None)
        if func_name is not None:
            getattr(self, func_name)(**kwargs)
        if current_level == 4 and self.move_away_recovery_counter < 3:
            self.move_away_recovery_counter += 1
            self.recovery_level_dict['obstacle'] = 2
        else:
            self.recovery_level_dict['obstacle'] = current_level + 1

    def recover_from_plan_failure(self, **kwargs):
        rospy.loginfo('Planning recovery')
        current_level = self.recovery_level_dict.get('plan')
        func_name = self._level_to_func.get(current_level, None)
        if func_name is not None:
            getattr(self, func_name)(**kwargs)
        self.recovery_level_dict['plan'] = current_level+1

    def recover_from_sync_failure(self, **kwargs):
        rospy.loginfo('Sync recovery')
        global_navigation_obj = kwargs.get('global_navigation_obj')
        feedback_msg = kwargs.get('feedback_msg')
        msg_wp = Utils.get_x_y_theta_from_pose(feedback_msg.reached_wp)
        geometric_path_wp = Utils.get_x_y_theta_from_pose(global_navigation_obj.geometric_path[0].pose)
        dist = Utils.get_distance_between_points(msg_wp[:2], geometric_path_wp[:2])
        # for i, wp in enumerate(global_navigation_obj.geometric_path):
        #     pose = Utils.get_x_y_theta_from_pose(wp.pose)
        #     print(pose)
        if dist > global_navigation_obj.geometric_wp_goal_tolerance:
            print('Robot has reached a wp that is different from path')
            # preempt and replan
            global_navigation_obj._cancel_bn_pub.publish(Empty())
            global_navigation_obj.geometric_path = None
            global_navigation_obj.choose_bn_mode()
        else:
            global_navigation_obj.geometric_path.pop(0)
            msg_remaining_path_length = feedback_msg.remaining_path_length
            nav_remaining_path_length = len(global_navigation_obj.geometric_path)
            if nav_remaining_path_length > msg_remaining_path_length:
                # basic navigation must have replaced instead of appending
                for i, wp in enumerate(global_navigation_obj.geometric_path):
                    pose = Utils.get_x_y_theta_from_pose(wp.pose)
                    if Utils.get_distance_between_points(msg_wp[:2], pose[:2]) > 0.1:
                        break
                for _ in range(i):
                    global_navigation_obj.geometric_path.pop(0)
            else:
                # preempt and replan
                global_navigation_obj._cancel_bn_pub.publish(Empty())
                global_navigation_obj.geometric_path = None
                global_navigation_obj.choose_bn_mode()
        current_level = self.recovery_level_dict.get('sync')
        self.recovery_level_dict['sync'] = current_level+1

    def _wait_recovery(self, **kwargs):
        rospy.loginfo('Recovery bahaviour: WAITING (duration: ' +
                      str(self.recovery_wait_duration) + ' seconds)')
        rospy.sleep(self.recovery_wait_duration)
        global_navigation_obj = kwargs.get('global_navigation_obj')
        if global_navigation_obj.geometric_path is not None:
            path_msg = Path()
            path_msg.header.stamp = rospy.Time.now()
            path_msg.header.frame_id = global_navigation_obj.global_frame
            path_msg.poses = global_navigation_obj.geometric_path
            global_navigation_obj._bn_path_pub.publish(path_msg)

    def _replan_recovery(self, **kwargs):
        rospy.loginfo('Recovery bahaviour: REPLAN')
        global_navigation_obj = kwargs.get('global_navigation_obj')
        global_navigation_obj.geometric_path = None
        global_navigation_obj.choose_bn_mode()

    def _reconfigure_recovery(self, **kwargs):
        rospy.loginfo('Recovery bahaviour: RECONFIGURE')
        global_navigation_obj = kwargs.get('global_navigation_obj')
        global_navigation_obj.geometric_planner.laser_utils.set_footprint_padding(0.02)
        global_navigation_obj.geometric_path = None
        global_navigation_obj._bn_mode_pub.publish(String(data='strict'))

    def _move_away_recovery(self, **kwargs):
        rospy.loginfo('Recovery bahaviour: MOVE AWAY FROM OBSTACLE (duration: ' +
                      str(self.recovery_motion_duration) + ' seconds)')
        direction = kwargs.get('direction', 3.14)
        x_vel = math.cos(direction) * self.recovery_vel
        y_vel = math.sin(direction) * self.recovery_vel
        recovery_twist = Utils.get_twist(x=x_vel, y=y_vel)
        start_time = rospy.get_time()
        while rospy.get_time() < start_time + self.recovery_motion_duration:
            rospy.sleep(0.1)
            self._cmd_vel_pub.publish(recovery_twist)

        global_navigation_obj = kwargs.get('global_navigation_obj')
        global_navigation_obj.geometric_planner.laser_utils.set_footprint_padding(self.default_footprint_padding)
        global_navigation_obj.geometric_path = None
        global_navigation_obj.choose_bn_mode()

    def _modify_goal_recovery(self, **kwargs):
        rospy.loginfo('Recovery bahaviour: MODIFY GOAL')
        global_navigation_obj = kwargs.get('global_navigation_obj')
        curr_pos = global_navigation_obj.get_current_position_from_tf()
        goal_pose = global_navigation_obj.topological_path[0]
        straight_line_path = global_navigation_obj.geometric_planner.plan_straight_line_path(
                                curr_pos, goal_pose[:3])
        plan = None
        while len(straight_line_path) > 0 and plan is None:
            goal_pose = straight_line_path.pop(-1)
            plan = global_navigation_obj.geometric_planner.plan(curr_pos, goal_pose)

        if plan is None:
            rospy.logerr('Modifying goal recovery failed.')
            return

        rospy.logwarn('Modifing current goal to make it achievable')
        global_navigation_obj.topological_path[0][:2] = goal_pose

    def _failure(self, **kwargs):
        rospy.logerr('Reached highest level of recovery. ABORTING')
        global_navigation_obj = kwargs.get('global_navigation_obj')
        global_navigation_obj._reset_state()
