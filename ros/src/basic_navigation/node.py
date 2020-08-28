class Node(object):

    """Class representing topology node of OSM areas"""

    def __init__(self, node_id, x, y, area_name, area_type):
        self.id = node_id
        self.x = round(x, 2)
        self.y = round(y, 2)
        self.area_name = area_name.encode('utf-8')
        self.area_type = area_type.encode('utf-8')

    def __repr__(self):
        string = "<"
        string += "id: " + str(self.id) + ", "
        string += "name: " + str(self.area_name) + ", "
        string += "type: " + str(self.area_type) + ", "
        string += "x: " + str(self.x) + ", "
        string += "y: " + str(self.y)
        string += ">"
        return string

    def __str__(self):
        return self.__repr__()

    def to_dict(self):
        dict_obj = {'id': self.id, 'x':self.x, 'y':self.y,
                    'area_name':self.area_name, 'area_type':self.area_type}
        return dict_obj

    @staticmethod
    def from_dict(dict_obj):
        """Initialise Node obj from dictionary

        :dict_obj: dict
        :returns: Node

        """
        return Node(dict_obj['id'], dict_obj['x'], dict_obj['y'], dict_obj['area_name'], dict_obj['area_type'])
