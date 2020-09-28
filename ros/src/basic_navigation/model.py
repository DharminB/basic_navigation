import tf
from utils import Utils

class Model(object):

    """Class representing model for MPC"""

    def __init__(self, acc_x=0.1, acc_y=0.1, acc_theta=0.1, default_control_time=1.0):
        self.acc_x = acc_x
        self.acc_y = acc_x
        self.acc_theta = acc_theta
        self.default_control_time = default_control_time

    def get_u_options(self):
        return [
                (0.0,           0.0,            0.0),
                (self.acc_x,    0.0,            0.0),
                (0.0,           self.acc_y,     0.0),
                (0.0,           0.0,            self.acc_theta),
                (-self.acc_x,   0.0,            0.0),
                (0.0,           -self.acc_y,    0.0),
                (0.0,           0.0,            -self.acc_theta),
               ]

    def get_trajectory(self, current_vel, acc_list, time_list=None):
        if time_list is None:
            time_list = [self.default_control_time for _ in range(len(acc_list))]
        trajectory = []
        pos = (0.0, 0.0, 0.0)
        vel = list(current_vel)
        for acc, t in zip(acc_list, time_list):
            vel[0] += acc[0]
            vel[1] += acc[1]
            vel[2] += acc[2]
            pos = Model.get_future_pose(pos[0], pos[1], pos[2],
                                        vel[0], vel[1], vel[2], t)
            trajectory.append((pos, tuple(vel)))
        return trajectory

    @staticmethod
    def get_future_pose(pos_x, pos_y, pos_theta, vel_x, vel_y, vel_theta, future_time):
        delta_t = 0.1
        vel_mat = tf.transformations.translation_matrix([vel_x*delta_t, vel_y*delta_t, 0.0])
        rot_mat = tf.transformations.euler_matrix(0.0 ,0.0, vel_theta*delta_t)
        vel_mat[:2, :2] = rot_mat[:2, :2]
        pos_mat = tf.transformations.translation_matrix([pos_x, pos_y, 0.0])
        pos_rot_mat = tf.transformations.euler_matrix(0.0 ,0.0, pos_theta)
        pos_mat[:2, :2] = pos_rot_mat[:2, :2]
        for i in range(int(future_time*10)):
            pos_mat = pos_mat.dot(vel_mat)
        _, _, theta = tf.transformations.euler_from_matrix(pos_mat)
        position = tf.transformations.translation_from_matrix(pos_mat)
        return (position[0], position[1], theta)

