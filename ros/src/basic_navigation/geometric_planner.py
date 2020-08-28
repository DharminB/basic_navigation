from __future__ import print_function

import math
import rospy

from utils import Utils
from laser_utils import LaserUtils
from nav_msgs.msg import Path

class GeometricPlanner(object):

    """Planner for short distance navigation using geometric sensor information"""

    def __init__(self, tf_listener=None, debug=False):
        # ros params
        self.global_frame = rospy.get_param('~global_frame', 'map')
        self.robot_frame = rospy.get_param('~robot_frame', 'load/base_link')
        self.dist_between_wp = rospy.get_param('~dist_between_wp', 1.0)
        self.connector_length = rospy.get_param('~connector_length', 0.5)
        footprint_padding = rospy.get_param('~footprint_padding', 0.1)

        # class variables
        self.tf_listener = tf_listener
        self.laser_utils = LaserUtils(footprint_padding=footprint_padding,
                                      debug=False, only_use_half=True)
        self.laser_utils.use_front_half = True

        # publishers
        if debug:
            self.path_debug_pub = rospy.Publisher('~debug_poses', Path, queue_size=1)

        rospy.sleep(0.2)

    def plan(self, start=(0.0, 0.0, 0.0), goal=(0.0, 0.0, 0.0), try_spline_first=False):
        """
        ASSUMPTION: start is current robot position
        ASSUMPTION: start and goal are in global frame

        :start: tuple of 3 float
        :goal: tuple of 3 float
        :try_spline_first: bool (used for junctions to create a curved turning path)
        :returns list of tuples (float, float, float)

        """
        if try_spline_first and Utils.get_distance_between_points(start[:2], goal[:2]) < 4.0 and abs(start[2]-goal[2]) > 0.2:
            rospy.loginfo('Trying spline junction first')
            path = self.plan_spline_path(start, goal, mode='overtake')
            # self.path_debug_pub.publish(Utils.get_path_msg_from_poses(path, self.global_frame))
            if self.is_path_safe(path)[0]:
                return path

        # first try straight line path
        straight_line_path = self.plan_straight_line_path(start, goal)

        straight_line_safe, collision_index = self.is_path_safe(straight_line_path)

        if straight_line_safe:
            return straight_line_path

        rospy.loginfo('Straight was not safe. Failed at '+ str(collision_index))

        # then try sampling at collision point and try path via a valid sample
        collision_pose_raw = straight_line_path[collision_index]
        transformed_collision = Utils.transform_pose(self.tf_listener, collision_pose_raw,
                                                     self.global_frame, self.robot_frame)
        transformed_collision = list(transformed_collision)
        transformed_collision[0] += self.laser_utils.get_footprint_edge_to_base_link_dist()
        collision_pose = Utils.transform_pose(self.tf_listener, transformed_collision,
                                              self.robot_frame, self.global_frame)
        theta = math.atan2(goal[1] - start[1], goal[0] - start[0])
        perpendicular_angle = Utils.get_perpendicular_angle(theta)

        for range_dist in range(2, 4):
            samples = self.generate_valid_samples_along_line(start, collision_pose,
                                                             perpendicular_angle,
                                                             lower_range=-float(range_dist),
                                                             higher_range=float(range_dist))
            if len(samples) > 0:
                break

        if len(samples) == 0:
            return None

        samples.sort(key=lambda sample: Utils.get_distance_between_points(collision_pose[:2], sample[:2]))
        self.path_debug_pub.publish(Utils.get_path_msg_from_poses(samples, self.global_frame))

        safe_wp_index = 1
        safe_path_found = False
        last_chance = False
        while not safe_path_found:
            if len(samples) > safe_wp_index:
                safe_wp = samples[safe_wp_index] 
            else:
                safe_wp = samples[0]
                last_chance = True
            x = (self.connector_length * math.cos(theta)) + safe_wp[0]
            y = (self.connector_length * math.sin(theta)) + safe_wp[1]
            safe_wp_2 = [x, y, theta]
            rospy.logdebug(safe_wp_index)
            first_half_path = self.plan_spline_path(start, safe_wp, mode='overtake')
            second_half_path = self.plan_spline_path(safe_wp_2, goal, mode='overtake')
            path = first_half_path
            path.extend(second_half_path)
            safe_path_found, _ = self.is_path_safe(path)
            if not safe_path_found:
                safe_wp_index += 1
                if last_chance:
                    return None
        return path

    def plan_straight_line_path(self, start, goal):
        diff_x = goal[0] - start[0]
        diff_y = goal[1] - start[1]
        dist = Utils.get_distance(diff_x, diff_y)
        theta = math.atan2(diff_y, diff_x)
        num_of_wp = int(math.floor(dist/self.dist_between_wp))
        path = []
        for i in range(num_of_wp):
            wp = [0.0, 0.0, 0.0]
            wp[0] = start[0] + (diff_x/num_of_wp * i)
            wp[1] = start[1] + (diff_y/num_of_wp * i)
            wp[2] = theta
            path.append(wp)
        path.append(goal)
        return path

    def plan_spline_path(self, start, goal, mode='overtake'):
        control_points = []
        angle_diff_raw = abs(start[2]-goal[2])
        angle_diff = angle_diff_raw - (math.pi*2) if angle_diff_raw > math.pi else angle_diff_raw
        if mode == 'overtake' and abs(angle_diff) < 0.5:
            # get 2 control points
            theta = goal[2]
            new_goal = [goal[0]-start[0], goal[1]-start[1], theta]
            rotated_new_goal = Utils.get_rotated_point(new_goal[:2], -theta)
            rotated_point_1 = (rotated_new_goal[0]/2, 0.0)
            rotated_point_2 = (rotated_new_goal[0]/2, rotated_new_goal[1])
            point_1 = list(Utils.get_rotated_point(rotated_point_1, theta))
            point_2 = list(Utils.get_rotated_point(rotated_point_2, theta))
            control_points.append((point_1[0]+start[0], point_1[1]+start[1]))
            control_points.append((point_2[0]+start[0], point_2[1]+start[1]))
        else: # use junction mode
            # get single control point
            theta = goal[2]
            new_goal = [goal[0]-start[0], goal[1]-start[1], theta]
            rotated_new_goal = Utils.get_rotated_point(new_goal[:2], -theta)
            rotated_point = (0.0, rotated_new_goal[1])
            point = list(Utils.get_rotated_point(rotated_point, theta))
            control_points.append((point[0]+start[0], point[1]+start[1]))

        control_points.insert(0, (start[0], start[1]))
        control_points.append((goal[0], goal[1]))
        n = Utils.calc_heuristic_n_from_points(control_points)
        curve_points = Utils.get_spline_curve(control_points, n=n)
        # path = [(point[0], point[1], 0.0) for point in control_points]
        path = []
        for i in range(len(curve_points)):
            if i < len(curve_points)-1:
                theta = math.atan2(curve_points[i+1][1] - curve_points[i][1],
                                   curve_points[i+1][0] - curve_points[i][0])
            else:
                theta = goal[2]
            path.append((curve_points[i][0], curve_points[i][1], theta))
        return path

    def generate_valid_samples_along_line(self, start, point, angle, dist_between_sample=0.2,
                                    lower_range=-3.0, higher_range=3.0):
        diff_x = math.cos(angle)
        diff_y = math.sin(angle)
        samples = []
        i = lower_range
        while i < higher_range:
            x = point[0] + (i * diff_x)
            y = point[1] + (i * diff_y)
            samples.append([x, y, point[2]])
            i += dist_between_sample

        safe_samples = []
        for sample in samples:
            transformed_sample = Utils.transform_pose(self.tf_listener, sample,
                                                      self.global_frame, self.robot_frame)
            safe = self.laser_utils.is_safe_from_colliding_at(*transformed_sample)
            if safe:
                straight_line_path = self.plan_straight_line_path(start, sample)
                safe_path, _ = self.is_path_safe(straight_line_path)
                if safe_path:
                    safe_samples.append(sample)

        return safe_samples

    def is_path_safe(self, path):
        for i, wp in enumerate(path[1:]):
            transformed_wp = Utils.transform_pose(self.tf_listener, wp,
                                                  self.global_frame, self.robot_frame)
            safe = self.laser_utils.is_safe_from_colliding_at(*transformed_wp)
            if not safe:
                return (False, i)
        return (True, -1)
