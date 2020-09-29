from __future__ import print_function

import tf
import random
import copy
import math
import time

from utils import Utils

class Optimiser(object):

    """Optimisation functions to use in MPC nav"""

    @staticmethod
    def calc_optimal_u_brute_force(model,
                                   goal,
                                   current_vel=[0.0, 0.0, 0.0],
                                   control_horizon=2,
                                   prediction_horizon=2,
                                   constraints=None):
        """
        Brute force method to find optimal u

        :goal: tuple of tuples ((float, float, float), ...)
        :control_horizon: int (positive)
        :prediction_horizon: int (positive)
        :returns: float, list of tuples [(float, float, float), ...]

        """
        if prediction_horizon < control_horizon:
            prediction_horizon = control_horizon
        u_options = model.get_u_options()
        indexes = [0 for i in range(control_horizon)]
        indexes[-1] = -1
        min_cost = float('inf')
        optimal_u = None
        while indexes.count(len(u_options)-1) < len(indexes):
            indexes[-1] += 1
            ind = len(indexes)-1
            while ind > 0 and indexes[ind] > len(u_options)-1:
                indexes[ind] = 0
                ind -= 1
                indexes[ind] += 1
            u = [u_options[i] for i in indexes]
            cost = Optimiser.calc_cost(model, u, current_vel, goal,
                                       control_horizon, prediction_horizon,
                                       constraints)
            if cost < min_cost:
                min_cost = cost
                optimal_u = copy.deepcopy(u)
            print(indexes)
            print(u)
            print(cost)
            print()
        print()
        print(min_cost)
        print(optimal_u)
        return min_cost, optimal_u

    @staticmethod
    def calc_optimal_u_gradient_descent_neighbour(model,
                                                  goal,
                                                  current_vel=[0.0, 0.0, 0.0],
                                                  control_horizon=2,
                                                  prediction_horizon=2,
                                                  initial_u=None,
                                                  constraints=None):
        """
        Gradient descent from initial_u

        :goal: tuple of tuples ((float, float, float), ...)
        :control_horizon: int (positive)
        :prediction_horizon: int (positive)
        :initial_u: list of tuple [(float, float, float), ...]
        :returns: float, list of tuples [(float, float, float), ...]

        """
        if prediction_horizon < control_horizon:
            prediction_horizon = control_horizon

        u_options = model.get_u_options()

        if initial_u is None:
            current = [0 for i in range(control_horizon)]
            u = [u_options[i] for i in current]
        else:
            current = [u_options.index(i) for i in initial_u]
            u = copy.deepcopy(initial_u)

        current_cost = Optimiser.calc_cost(model, u, current_vel, goal,
                                           control_horizon, prediction_horizon,
                                           constraints)
        print(current, current_cost)
        for _ in range(10):
            # generate neighbours
            neighbours = []
            for i in range(control_horizon):
                for j in range(len(u_options)):
                    if current[i] != j:
                        neighbour = copy.deepcopy(current)
                        neighbour[i] = j
                        neighbours.append(neighbour)

            # find costs of neighbours
            min_cost = float('inf')
            min_cost_neighbour = neighbours[0]
            for neighbour in neighbours:
                u = [u_options[i] for i in neighbour]
                cost = Optimiser.calc_cost(model, u, current_vel, goal,
                                           control_horizon, prediction_horizon,
                                           constraints)
                # print(neighbour, cost)
                if cost < min_cost:
                    min_cost = cost
                    min_cost_neighbour = copy.deepcopy(neighbour)

            # make current equal to neighbour with min cost if better
            if min_cost > current_cost:
                break

            current = min_cost_neighbour
            current_cost = min_cost
            # print(current, current_cost)
            # print()
        # print()
        optimal_u = [u_options[i] for i in current]
        # print(optimal_u, current_cost)
        return current_cost, optimal_u

    @staticmethod
    def calc_optimal_u_gradient_descent(model,
                                        goal,
                                        current_vel=[0.0, 0.0, 0.0],
                                        control_horizon=2,
                                        prediction_horizon=2,
                                        initial_u=None,
                                        constraints=None):
        """
        Gradient descent from initial_u

        :model: Model object
        :goal: tuple of tuples ((float, float, float), ...)
        :current_vel: list of float [float, float, float]
        :control_horizon: int (positive)
        :prediction_horizon: int (positive)
        :initial_u: list of tuple [(float, float, float), ...]
        :constraints: dict
        :returns: float, list of tuples [(float, float, float), ...]

        """
        if prediction_horizon < control_horizon:
            prediction_horizon = control_horizon

        h = 0.0001
        alpha = 0.00001
        cost_threshold = 0.1

        if initial_u is None:
            u = [[0.0, 0.0, 0.0] for _ in range(control_horizon)]
        else:
            u = [list(i) for i in initial_u]

        current_cost = Optimiser.calc_cost(model, u, current_vel, goal,
                                           control_horizon, prediction_horizon,
                                           constraints)
        Optimiser.print_u_and_cost(u, current_cost)
        itr_num = 0
        while True:
            itr_num += 1
            # calculate gradient
            gradient = Optimiser.calc_gradient(model, goal, current_vel,
                            control_horizon, prediction_horizon, constraints,
                            h, u, current_cost)

            # Optimiser.print_u_and_cost(gradient, 0.0, 6)

            for i in range(control_horizon):
                for j in range(3):
                    u[i][j] -= alpha * gradient[i][j]
                    u[i][j] = Utils.clip(u[i][j], constraints['max_acc'][j],
                                         constraints['min_acc'][j])

            new_cost = Optimiser.calc_cost(model, u, current_vel, goal,
                                           control_horizon, prediction_horizon,
                                           constraints)
            delta_cost = current_cost - new_cost
            current_cost = new_cost
            Optimiser.print_u_and_cost(u, current_cost)
            print(itr_num, current_cost)
            if abs(delta_cost) < cost_threshold:
                break

        return current_cost, u

    @staticmethod
    def calc_optimal_u_line_search(model,
                                   goal,
                                   current_vel=[0.0, 0.0, 0.0],
                                   control_horizon=2,
                                   prediction_horizon=2,
                                   initial_u=None,
                                   constraints=None):
        """
        Line search from initial_u

        :model: Model object
        :goal: tuple of tuples ((float, float, float), ...)
        :current_vel: list of float [float, float, float]
        :control_horizon: int (positive)
        :prediction_horizon: int (positive)
        :initial_u: list of tuple [(float, float, float), ...]
        :constraints: dict
        :returns: float, list of tuples [(float, float, float), ...]

        """
        if prediction_horizon < control_horizon:
            prediction_horizon = control_horizon

        h = 0.0001
        cost_threshold = 0.1

        if initial_u is None:
            u = [[0.0, 0.0, 0.0] for _ in range(control_horizon)]
        else:
            u = [list(i) for i in initial_u]

        current_cost = Optimiser.calc_cost(model, u, current_vel, goal,
                                           control_horizon, prediction_horizon,
                                           constraints)
        Optimiser.print_u_and_cost(u, current_cost)
        start_time = time.time()
        itr_num = 0
        while True:
            itr_num += 1
            # calculate gradient
            gradient = Optimiser.calc_gradient(model, goal, current_vel,
                            control_horizon, prediction_horizon, constraints,
                            h, u, current_cost)

            # Optimiser.print_u_and_cost(gradient, 0.0, 6)

            # calculate new u with "loosely" optimal alpha
            min_cost, new_u = Optimiser.calc_new_u_with_optimal_alpha(model, goal,
                                current_vel, control_horizon, prediction_horizon,
                                constraints, u, current_cost, gradient)
            delta_cost = current_cost - min_cost
            current_cost = min_cost
            u = new_u
            # Optimiser.print_u_and_cost(u, current_cost)
            print(itr_num, time.time() - start_time, current_cost)
            if abs(delta_cost) < cost_threshold:
                break

        return current_cost, u

    @staticmethod
    def calc_optimal_u_conjugate_gradient_descent(model,
                                   goal,
                                   current_vel=[0.0, 0.0, 0.0],
                                   control_horizon=2,
                                   prediction_horizon=2,
                                   initial_u=None,
                                   constraints=None):
        """
        Conjugate gradient descent from initial_u
        Conjugate direction using Polak-Ribiere

        :model: Model object
        :goal: tuple of tuples ((float, float, float), ...)
        :current_vel: list of float [float, float, float]
        :control_horizon: int (positive)
        :prediction_horizon: int (positive)
        :initial_u: list of tuple [(float, float, float), ...]
        :constraints: dict
        :returns: float, list of tuples [(float, float, float), ...]

        """
        if prediction_horizon < control_horizon:
            prediction_horizon = control_horizon

        h = 0.0001
        cost_threshold = 0.1

        if initial_u is None:
            u = [[0.0, 0.0, 0.0] for _ in range(control_horizon)]
        else:
            u = [list(i) for i in initial_u]

        current_cost = Optimiser.calc_cost(model, u, current_vel, goal,
                                           control_horizon, prediction_horizon,
                                           constraints)
        Optimiser.print_u_and_cost(u, current_cost)
        prev_steepest_direction = None
        prev_conjugate_direction = None
        start_time = time.time()
        itr_num = 0
        while True:
            itr_num += 1
            # print(itr_num)
            # calculate steepest direction
            steepest_direction = Optimiser.calc_gradient(model, goal, current_vel,
                            control_horizon, prediction_horizon, constraints,
                            h, u, current_cost)
            for i in range(control_horizon):
                for j in range(3):
                    steepest_direction[i][j] *= -1

            # Optimiser.print_u_and_cost(steepest_direction, 0.0, 6)

            # calculate conjugate direction
            if prev_steepest_direction is None and prev_conjugate_direction is None:
                conjugate_direction = copy.deepcopy(steepest_direction)
            else:
                numerator = 0.0
                denominator = 0.0
                for i in range(control_horizon):
                    for j in range(3):
                        numerator += steepest_direction[i][j] *\
                                (steepest_direction[i][j]-prev_steepest_direction[i][j])
                        denominator += prev_steepest_direction[i][j]**2
                beta = max(0.0, numerator/denominator)
                # print(beta)

                conjugate_direction = copy.deepcopy(steepest_direction)
                for i in range(control_horizon):
                    for j in range(3):
                        conjugate_direction[i][j] += beta * prev_conjugate_direction[i][j]
            prev_steepest_direction = copy.deepcopy(steepest_direction)
            prev_conjugate_direction = copy.deepcopy(conjugate_direction)

            # Optimiser.print_u_and_cost(conjugate_direction, 0.0, 6)

            # calculate new u with "loosely" optimal alpha
            for i in range(control_horizon):
                for j in range(3):
                    conjugate_direction[i][j] *= -1
            min_cost, new_u = Optimiser.calc_new_u_with_optimal_alpha(model, goal,
                                current_vel, control_horizon, prediction_horizon,
                                constraints, u, current_cost, conjugate_direction)

            delta_cost = current_cost - min_cost
            current_cost = min_cost
            u = new_u
            # Optimiser.print_u_and_cost(u, current_cost)
            print(itr_num, time.time() - start_time, current_cost)
            if abs(delta_cost) < cost_threshold:
                break

        return current_cost, u

    @staticmethod
    def calc_optimal_u_gradient_descent_random_restart(model,
                                                       goal,
                                                       current_vel=[0.0, 0.0, 0.0],
                                                       control_horizon=2,
                                                       prediction_horizon=2,
                                                       num_of_restart=5,
                                                       constraints=None):
        """
        Calculate optimal u using gradient descent with random restart

        """
        u_options = model.get_u_options()
        min_cost = float('inf')
        optimal_u = None
        for _ in range(num_of_restart):
            initial_u = [random.choice(u_options) for _ in range(control_horizon)]
            print(initial_u)
            cost, u = Optimiser.calc_optimal_u_gradient_descent_neighbour(
                                model, goal, current_vel, control_horizon,
                                prediction_horizon, initial_u, constraints)
            if cost < min_cost:
                min_cost = cost
                optimal_u = copy.deepcopy(u)
            print(cost, u)
        # print()
        # print(optimal_u, min_cost)
        return min_cost, optimal_u

    @staticmethod
    def calc_new_u_with_optimal_alpha(model, goal, current_vel, control_horizon,
                      prediction_horizon, constraints, u, current_cost, gradient):
        min_cost = current_cost
        best_new_u = copy.deepcopy(u)
        alpha = 0.000001
        while alpha <= 1.0:
            new_u = copy.deepcopy(u)
            for i in range(control_horizon):
                for j in range(3):
                    new_u[i][j] -= alpha * gradient[i][j]
                    u[i][j] = Utils.clip(u[i][j], constraints['max_acc'][j],
                                         constraints['min_acc'][j])

            new_cost = Optimiser.calc_cost(model, new_u, current_vel, goal,
                                           control_horizon, prediction_horizon,
                                           constraints)
            # print(alpha, new_cost)
            # Optimiser.print_u_and_cost(new_u, new_cost)
            if new_cost < min_cost:
                min_cost = new_cost
                best_new_u = new_u
                alpha *= 10.0
            else:
                break
        return min_cost, best_new_u

    @staticmethod
    def calc_gradient(model, goal, current_vel, control_horizon,
                      prediction_horizon, constraints, h, u, current_cost):
        gradient = [[0.0, 0.0, 0.0] for _ in range(control_horizon)]
        for i in range(control_horizon):
            for j in range(3):
                neighbour = copy.deepcopy(u)
                neighbour[i][j] += h
                neighbour_cost = Optimiser.calc_cost(
                                    model, neighbour, current_vel, goal,
                                    control_horizon, prediction_horizon,
                                    constraints)
                gradient[i][j] = (neighbour_cost - current_cost)/h
        return gradient

    @staticmethod
    def calc_cost(model, u, current_vel, goal, control_horizon, prediction_horizon, constraints):
        # add cruise control between control horizon and prediction horizon
        control_input = copy.deepcopy(u)
        for _ in range(prediction_horizon-control_horizon):
            control_input.append((0.0, 0.0, 0.0))

        # generate trajectory using model
        trajectory = model.get_trajectory(current_vel, control_input)

        cost = 0.0

        if constraints is not None:
            future_velocities = [i[1] for i in trajectory]
            future_poses = [i[0] for i in trajectory]

            # vel limit soft constraints
            if 'max_vel' in constraints and 'min_vel' in constraints:
                for vel in future_velocities:
                    for i in range(3):
                        vel_diff_max = vel[i] - constraints['max_vel'][i]
                        vel_diff_min = constraints['min_vel'][i] - vel[i]
                        cost += 1000.0 * max(0.0, vel_diff_max, vel_diff_min)**2

            # collision soft constraints
            if 'points' in constraints:
                for pose in future_poses:
                    cost += 100.0 * Optimiser.get_laser_points_cost_at(constraints['points'], *pose)

        # cost for reaching goal state
        horizon_state = trajectory[-1]
        # linear position error
        cost += 10.0 * Utils.get_distance_between_points(horizon_state[0][:2], goal[0][:2])**2
        # angular position error
        cost += 10.0 * Utils.get_shortest_angle(horizon_state[0][2], goal[0][2])**2
        # linear velocity error
        cost += 10.0 * Utils.get_distance_between_points(horizon_state[1][:2], goal[1][:2])**2
        # angular velocity error
        cost += 10.0 * (horizon_state[1][2]-goal[1][2])**2

        return cost

    @staticmethod
    def get_laser_points_cost_at(points, x=0.0, y=0.0, theta=0.0):
        robot_radius = 0.5 # FIXME
        base = 0.1
        exp_multiplier = 5.0
        robot_pt = (x, y)
        cost = 0.0
        for pt in points:
            dist = Utils.get_distance_between_points(robot_pt, pt)
            value = base**(exp_multiplier*(dist-robot_radius))
            cost += value
            # print(dist, value)
        # print(cost)
        # print()
        return cost

    @staticmethod
    def print_u_and_cost(u, cost, rounding_num=3):
        print("-"*20)
        for acc in u:
            print([round(acc[0], rounding_num), round(acc[1], rounding_num), round(acc[2], rounding_num)])
        print(round(cost, rounding_num))
        print("-"*20)
