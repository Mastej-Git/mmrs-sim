class Resource:

    def __init__(self, resource_id, path_id_1: int, path_id_2: int):
        self.id = resource_id
        self.paths = (path_id_1, path_id_2)
        self.priority_list = []

    def get_access(self, robot_id):
        if robot_id not in self.priority_list:
            self.priority_list.append(robot_id)

    def release(self, robot_id):
        if robot_id in self.priority_list:
            self.priority_list.remove(robot_id)

    def is_first(self, robot_id):
        return len(self.priority_list) > 0 and self.priority_list[0] == robot_id
        