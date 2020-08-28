from __future__ import print_function

import yaml
from node import Node
from utils import Utils
from geometry_msgs.msg import Point

class TopologicalPlanner(object):

    """Planner for topological nodes"""

    def __init__(self, network_file):
        with open(network_file, 'r') as file_obj:
            network = yaml.safe_load(file_obj)
        self.nodes = {node['id']:Node.from_dict(node) for node in network['nodes']}
        self._initialise_neighbours(network['connections'])

    def _initialise_neighbours(self, connections):
        self.neighbours = {}
        for node_id in self.nodes:
            neighbour_list = []
            for connection in connections:
                if node_id in connection:
                    neighbour_list.append(connection[0] if node_id == connection[1] else connection[1])
            self.neighbours[node_id] = neighbour_list

    def plan(self, start=(0.0, 0.0), goal=(0.0, 0.0)):
        """Plan a path from start point to goal point

        :start: tuple of 2 float
        :goal: tuple of 2 float
        :returns: list of Node or None

        """
        start_node = self.get_nearest_topological_point(*start)
        goal_node = self.get_nearest_topological_point(*goal)
        plan = self.plan_path(start_node, goal_node)

        if plan is None:
            return None

        start_point_node = Node(node_id=10000, x=start[0], y=start[1], area_name='', area_type='')
        goal_point_node = Node(node_id=10001, x=goal[0], y=goal[1], area_name='', area_type='')

        if len(plan) == 1:
            plan = [start_point_node, goal_point_node]
        else: # whether to keep first area and last area or not
            first_node = (plan[0].x, plan[0].y)
            second_node = (plan[1].x, plan[1].y)
            last_node = (plan[-1].x, plan[-1].y)
            last_second_node = (plan[-2].x, plan[-2].y)
            dist_1 = Utils.get_distance_between_points(start[:2], second_node)
            dist_2 = Utils.get_distance_between_points(first_node, second_node)
            if dist_1 < dist_2:
                plan.pop(0)
            dist_1 = Utils.get_distance_between_points(goal[:2], last_second_node)
            dist_2 = Utils.get_distance_between_points(last_node, last_second_node)
            if dist_1 < dist_2:
                plan.pop()
            plan.insert(0, start_point_node)
            plan.append(goal_point_node)
        return plan

    def plan_path(self, start_node, goal_node):
        """Plan a path from start node to goal

        :start_node: int
        :goal_node: int
        :returns: list of Node

        """
        topological_path = self.search(start_node, goal_node)
        if topological_path is None:
            return None

        node_path = [self.nodes[node_id] for node_id in topological_path]
        return node_path

    def get_nearest_topological_point(self, x, y):
        """
        Finds nearest topological point from network to the given (x, y) point

        :x: float
        :y: float
        :returns: int

        """
        nearest_node = self.nodes[self.nodes.keys()[0]]
        min_dist = float('inf')
        for node in list(self.nodes.values()):
            dist = Utils.get_distance_between_points((node.x, node.y), (x, y))
            if dist < min_dist:
                min_dist = dist
                nearest_node = node.id
        return nearest_node

    def search(self, start, goal):
        fringe = [(start, self.get_distance_between_nodes(start, goal))]
        visited = []
        parent = {}
        while len(fringe) > 0:
            curr_node, _ = fringe.pop(fringe.index(min(fringe, key=lambda n: n[1])))
            if curr_node == goal:
                topological_path = [goal]
                while topological_path[-1] in parent:
                    topological_path.append(parent[topological_path[-1]])
                return topological_path[::-1]

            visited.append(curr_node)
            for n in self.neighbours[curr_node]:
                if n not in visited:
                    fringe.append((n, self.get_distance_between_nodes(n, goal)))
                    parent[n] = curr_node

    def get_distance_between_nodes(self, n1, n2):
        point1 = (self.nodes[n1].x, self.nodes[n1].y)
        point2 = (self.nodes[n2].x, self.nodes[n2].y)
        return Utils.get_distance_between_points(point1, point2)
