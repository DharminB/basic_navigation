from __future__ import print_function

import tf
import math
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Pose, TransformStamped, Quaternion
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker, InteractiveMarker, InteractiveMarkerControl

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
    def get_perpendicular_angle(angle):
        """Compute the angle which is perpendicular to given angle and ensures that the 
        returned angle is between pi and -pi
        ASSUMPTION: angle is always between pi and -pi

        :angle: float
        :returns: float
        """
        perpendicular_angle = angle + math.pi/2
        if perpendicular_angle > math.pi:
            perpendicular_angle -= math.pi
        return perpendicular_angle

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
    def get_spline_curve(points, n=11):
        """Return `n` points on spline curve defined by `points` where first and 
        last point is start and end point and all the points in the middle are 
        control points.

        :points: list of tuple(float, float)
        :n: int
        :returns: list of tuple(float, float)

        """
        order = len(points) - 1
        coef = Utils.pascal_triangle_row(order)
        offset = 1.0 / (n-1)
        curve_points = []
        curve_points.append(points[0]) # add start point as first curve point
        # add n-2 curve points in middle
        for factor in range(1, n-1):
            t = offset * factor
            x, y = 0.0, 0.0
            for i in range(order+1):
                x += coef[i] * (1-t)**(order-i) * t**(i) * points[i][0]
                y += coef[i] * (1-t)**(order-i) * t**(i) * points[i][1]
            curve_points.append((x, y))
        curve_points.append(points[-1]) # add end point as last curve point
        return curve_points

    @staticmethod
    def pascal_triangle_row(n):
        """Returns `n`th row from pascal triangle.

        :n: int
        :returns: list of int

        """
        coef = [1]
        for i in range(1, n+1):
            coef.append(int(coef[-1] * ((n + 1.0 - i)/i)))
        return coef

    @staticmethod
    def calc_heuristic_n_from_points(points):
        dist = 0.0
        for i in range(len(points)-1):
            dist += Utils.get_distance_between_points(points[i], points[i+1])
        return max(int(dist*2.0), 5)

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

    @staticmethod
    def get_future_positions(vel_x, vel_theta, num_of_points, future_time):
        """
        Calculate a bunch of points where the robot would be (in base_link) in future when 
        certain velocity are executed.
        The last point would be the position of robot almost at `future_time` and first
        point is the robot's current position.

        :vel_x: float (forward linear velocity)
        :vel_theta: float (angular yaw velocity)
        :num_of_points: int (number of points to generate)
        :future_time: float (seconds)
        :returns: list of geometry_msgs.Point

        """
        dist = abs(vel_x) * future_time
        angular_dist = abs(vel_theta) * future_time
        radius = dist/angular_dist

        sign_x = 1 if vel_x > 0 else -1
        sign_theta = 1 if vel_theta > 0 else -1

        theta_inc = angular_dist/num_of_points
        points = []
        for i in range(num_of_points):
            theta = i * theta_inc
            x = sign_x * (radius * math.sin(theta))
            y = sign_theta * radius * (1 - math.cos(theta))
            points.append(Point(x=x, y=y, z=0.0))
        return points

    @staticmethod
    def get_future_poses(vel_x, vel_theta, num_of_points, future_time):
        """
        Calculate a bunch of poses(x, y, theta) where the robot would be 
        (in base_link) in future when certain velocity are executed.
        The last point would be the position of robot almost at `future_time` and first
        point is the robot's current position.

        :vel_x: float (forward linear velocity)
        :vel_theta: float (angular yaw velocity)
        :num_of_points: int (number of points to generate)
        :future_time: float (seconds)
        :returns: list of tuples (float, float, float)

        """
        vel_theta = max(vel_theta, 0.0001)
        dist = abs(vel_x) * future_time
        angular_dist = abs(vel_theta) * future_time
        radius = dist/angular_dist

        sign_x = 1 if vel_x > 0 else -1
        sign_theta = 1 if vel_theta > 0 else -1

        theta_inc = angular_dist/num_of_points
        points = []
        for i in range(num_of_points):
            theta = i * theta_inc
            x = sign_x * (radius * math.sin(theta))
            y = sign_theta * radius * (1 - math.cos(theta))
            points.append((x, y, theta))
        return points

    @staticmethod
    def get_path_msg_from_poses(poses, frame_id):
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = frame_id

        path_msg.poses = [Utils.get_pose_stamped_from_frame_x_y_theta(frame_id, pose[0], pose[1], pose[2])
                          for pose in poses]
        return path_msg

    @staticmethod
    def ray_tracing_algorithm(vertx, verty, testx, testy):
        """Checks if the point (textx, testy) is inside polygon defined by list vertx and verty
        Taken from : https://stackoverflow.com/a/2922778/10460994
        Implements ray tracing algorithm.

        :vertx: list of floats
        :verty: list of floats
        :testx: float
        :testy: float
        :returns: bool

        """
        j = -1
        counter = 0
        for i in range(len(vertx)):
            if (verty[i] > testy) != (verty[j] > testy) and \
               (testx < ((vertx[j] - vertx[i]) * ((testy - verty[i]) / (verty[j] - verty[i])) + vertx[i])):
                counter += 1
            j = i
        return (counter % 2 == 1)

    @staticmethod
    def transform_pose(listener, input_pose, current_frame, target_frame):
        """Wrapper on tf.TransformListener.transformPose to use pose in 2d (x, y, theta)

        :listener: tf.TransformListener
        :input_pose: tuple of 3 float
        :current_frame: string
        :target_frame: string
        :returns: tuple of 3 float

        """
        pose = Utils.get_pose_stamped_from_frame_x_y_theta(current_frame, *input_pose)
        try:
            common_time = listener.getLatestCommonTime(current_frame, target_frame)
            pose.header.stamp = common_time
            transformed_pose = listener.transformPose(target_frame, pose)
            return Utils.get_x_y_theta_from_pose(transformed_pose.pose)
        except Exception as e:
            rospy.logerr(str(e))
            return None

    @staticmethod
    def get_rotated_point(point, angle):
        """Rotate point with angle

        :point: tuple (float, float)
        :angle: float (between -pi and pi)
        :returns: tuple (float, float)

        """
        x = (math.cos(angle) * point[0]) + (-math.sin(angle) * point[1])
        y = (math.sin(angle) * point[0]) + (math.cos(angle) * point[1])
        return (x, y)

    @staticmethod
    def get_2_dof_interactive_marker(marker_name, frame, x=0.0, y=0.0):
        """Return an interactive marker with 2 degree of freedom (X and Y axis)
        in `frame` at (`x`, `y`, 0.0) position named `name`

        :marker_name: string
        :frame: string
        :x: int
        :y: int
        :returns: visualization_msgs.InteractiveMarker

        """
        # create an interactive marker for our server
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = frame
        int_marker.name = marker_name
        int_marker.pose.position.x = x
        int_marker.pose.position.y = y
        # int_marker.description = "Simple 2-DOF Control"

        # create a grey box marker
        box_marker = Marker()
        box_marker.type = Marker.SPHERE
        box_marker.scale.x = box_marker.scale.y = box_marker.scale.z = 0.1
        box_marker.color.r = box_marker.color.a = 1.0
        box_marker.color.g = box_marker.color.b = 0.0

        # create a non-interactive control which contains the box
        box_control = InteractiveMarkerControl()
        box_control.always_visible = True
        box_control.markers.append( box_marker )

        # add the control to the interactive marker
        int_marker.controls.append( box_control )

        # create a control which will move the box
        # this control does not contain any markers,
        # which will cause RViz to insert two arrows
        rotate_control = InteractiveMarkerControl()
        rotate_control.name = "move_x"
        rotate_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS

        # add the control to the interactive marker
        int_marker.controls.append(rotate_control);

        rotate_control2 = InteractiveMarkerControl()
        rotate_control2.orientation.z = rotate_control2.orientation.w = 0.707
        rotate_control2.name = "move_y"
        rotate_control2.interaction_mode = InteractiveMarkerControl.MOVE_AXIS

        # add the control to the interactive marker
        int_marker.controls.append(rotate_control2);
        return int_marker

    @staticmethod
    def get_path_length(poses):
        """Calculate the length of path defined by poses

        :poses: list of tuples (float, float [, float, ..])
        :returns: float

        """
        dist = 0.0
        for i in range(len(poses)-1):
            dist += Utils.get_distance_between_points(poses[i][:2], poses[i+1][:2])
        return dist

    @staticmethod
    def get_path_length_pose_stamped(pose_stamped_list):
        """Calculate the length of path defined by pose stamped msgs

        :pose_stamped_list list of geometry_msgs.PoseStamped
        :returns: float

        """
        poses = [Utils.get_x_y_theta_from_pose(p.pose) for p in pose_stamped_list]
        return Utils.get_path_length(poses)
