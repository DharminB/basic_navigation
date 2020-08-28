from __future__ import print_function

import yaml
from OBL import OSMBridge, PathPlanner
from node import Node

REF_LATITUDE = 50.7800401
REF_LONGITUDE = 7.18226
BUILDING = 'BRSU'

class OSMNetworkCreator(object):

    """Create a graph for an OSM floor at area level"""

    def __init__(self, lat, lon, building, floor):
        self.osm_bridge = OSMBridge(global_origin=[REF_LATITUDE, REF_LONGITUDE])
        self.floor = self.osm_bridge.get_floor(building + '_L' + str(floor))

    def build_graph(self):
        nodes = []
        node_ids = []
        connections = []
        connection_ids = self.floor.connection_ids
        connection_ids.extend(self.floor._member_ids['global_connection_blocked'] if 'global_connection_blocked' in self.floor._member_ids else [])
        for connection_id in connection_ids:
            connection = self.osm_bridge.get_connection(connection_id)
            connection_list = connection.point_ids

            if len(connection_list) == 2:
                connections.append(connection_list)
            else:
                for i in range(len(connection_list)-1):
                    connections.append([connection_list[i], connection_list[i+1]])

            for point in connection.points:
                node = self.get_node_from_point(point)
                if node.id not in node_ids:
                    node_ids.append(node.id)
                    nodes.append(node)
        print('Nodes found:', len(nodes))
        print('Edges found:', len(connections))
        self.write_in_yaml(nodes, connections)
        print('Writing info in /tmp/osm_network.yaml')

    def write_in_yaml(self, nodes, connections):
        with open('/tmp/osm_network.yaml', 'w') as file_obj:
            node_dict = [node.to_dict() for node in nodes]
            yaml.dump({'nodes':node_dict}, file_obj, default_flow_style=False)
            yaml.dump({'connections':connections}, file_obj, default_flow_style=True)

    def get_node_from_point(self, point):
        _, _, relations = point.osm_adapter.get_parent(
            id=point.id, data_type='node', parent_child_role='topology')
        relation = relations[0]
        area_name, area_type = None, None
        for tag in relations[0].tags:
            if tag.key == "type":
                area_type = tag.value
            if tag.key == "ref":
                area_name = tag.value
        n = Node(point.id, point.x, point.y, area_name, area_type)
        return n

if __name__ == "__main__":
    ONC = OSMNetworkCreator(REF_LATITUDE, REF_LONGITUDE, BUILDING, floor=0)
    ONC.build_graph()
