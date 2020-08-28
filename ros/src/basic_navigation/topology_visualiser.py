from __future__ import print_function

import yaml
import rospy
from interactive_markers.interactive_marker_server import InteractiveMarkerServer, InteractiveMarkerFeedback

from std_msgs.msg import String
from geometry_msgs.msg import Pose, PointStamped, Point
from visualization_msgs.msg import MarkerArray, Marker, InteractiveMarker, InteractiveMarkerControl

from node import Node
from utils import Utils

class TopologyVisualiser(object):

    """Visualise topology graph in rviz (and edit it)"""

    def __init__(self, network_file):
        self.network_file = network_file
        with open(network_file, 'r') as file_obj:
            network = yaml.safe_load(file_obj)
        self.nodes = {node['id']:Node.from_dict(node) for node in network['nodes']}
        self.connections = network['connections']
        self.node_id_counter = max(self.nodes.keys())

        # ros params
        self.frame = rospy.get_param('frame', 'map')

        self._topology_pub = rospy.Publisher('/topology', MarkerArray, queue_size=1)
        self.interactive_marker_server = InteractiveMarkerServer("two_dof_marker")

        rospy.Subscriber('/clicked_point', PointStamped, self.clicked_point_cb)
        rospy.Subscriber('~command', String, self.command_cb)

        rospy.sleep(1.0)

        self._topology_pub.publish(self.get_marker_array())
        rospy.sleep(1.0)

        for node in self.nodes.values():
            self._add_interactive_marker(node.x, node.y, node.id)

        rospy.loginfo('Initialised')

    def save(self):
        with open(self.network_file, 'w') as file_obj:
            node_dict = [node.to_dict() for node in self.nodes.values()]
            yaml.dump({'nodes':node_dict}, file_obj, default_flow_style=False)
            yaml.dump({'connections':self.connections}, file_obj, default_flow_style=False)

    def command_cb(self, msg):
        command = msg.data
        if 'add connection' in command or 'del connection' in command:
            command_list = command.split()
            if len(command_list) != 4:
                self.print_command_warning()
                return
            try:
                conn1 = int(command_list[2])
                conn2 = int(command_list[3])
            except Exception as e:
                self.print_command_warning()
                return
            if conn1 not in self.nodes:
                rospy.logwarn(command_list[2] + ' not in nodes list')
                return
            if conn1 not in self.nodes:
                rospy.logwarn(command_list[2] + ' not in nodes list')
                return
            if command_list[0] == 'add':
                self.connections.append([conn1, conn2])
                rospy.loginfo('Added connection')
            else:
                connection = [conn1, conn2]
                if connection in self.connections:
                    self.connections.remove(connection)
                elif connection[::-1] in self.connections:
                    self.connections.remove(connection[::-1])
                else:
                    rospy.logwarn(str(connection) + ' not in connections list')
                    return
                rospy.loginfo('Deleted connection')
            self._topology_pub.publish(self.get_marker_array())
            rospy.sleep(0.5)
        else:
            self.print_command_warning()

    def print_command_warning(self):
        rospy.logwarn('Invalid command format.')
        print('Valid commands are:')
        for i in ["add connection", "del connection"]:
            print('-', i)
        print('Examples:')
        print('- "add connection 123 234"')
        print('- "del connection 123 234"')

    def clicked_point_cb(self, msg):
        rospy.loginfo(msg)
        self.node_id_counter += 1
        node_id = self.node_id_counter
        self.nodes[node_id] = Node(node_id, msg.point.x, msg.point.y, "BRSU_area", "area")
        self._topology_pub.publish(self.get_marker_array())
        self._add_interactive_marker(msg.point.x, msg.point.y, node_id)
        rospy.loginfo('Added node successfully')
        rospy.sleep(0.5)

    def get_marker_array(self):
        """Return a MarkerArray object representing the graph formed by topology
        nodes and connections

        :returns: visualization_msgs.MarkerArray

        """
        marker_array = MarkerArray()
        for i, node in enumerate(self.nodes.values()):
            marker_msg_text = Marker(type=Marker.TEXT_VIEW_FACING, id=node.id)
            marker_msg_text.text = str(node.id)+" ("+str(node.area_type)+")"
            marker_msg_text.scale.z = 0.5
            marker_msg_text.color.a = 1.0
            marker_msg_text.pose.position.x = node.x
            marker_msg_text.pose.position.y = node.y
            marker_msg_text.header.stamp = rospy.Time.now()
            marker_msg_text.header.frame_id = self.frame
            marker_array.markers.append(marker_msg_text)

        for i, conn in enumerate(self.connections):
            start_node = self.nodes[conn[0]]
            end_node = self.nodes[conn[1]]
            marker = Marker(type=Marker.LINE_STRIP, id=10000+i)
            marker.points.append(Point(x=start_node.x, y=start_node.y))
            marker.points.append(Point(x=end_node.x, y=end_node.y))
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = self.frame
            marker.color.b = marker.color.a = 1.0
            marker.scale.x = 0.05
            marker_array.markers.append(marker)
        return marker_array

    def _add_interactive_marker(self, x, y, node_id):
        """adds a interactive marker at the position.

        :x: float
        :y: float
        :node_id: int or None
        :returns: None

        """
        marker = Utils.get_2_dof_interactive_marker(str(node_id), self.frame, x, y)
        self.interactive_marker_server.insert(marker, self.interactive_marker_cb)
        self.interactive_marker_server.applyChanges()

    def interactive_marker_cb(self, feedback):
        """Updates the control point when interactive marker is moved.

        :feedback: InteractiveMarkerFeedback
        :returns: None

        """
        if not feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            return
        p = feedback.pose.position
        node_id = int(feedback.marker_name)
        if node_id in self.nodes:
            self.nodes[node_id].x = round(p.x, 2)
            self.nodes[node_id].y = round(p.y, 2)
        else:
            rospy.logerr("Incorrect interactive marker id " + feedback.marker_name) # should never come here
        self._topology_pub.publish(self.get_marker_array())
        rospy.sleep(0.5)
