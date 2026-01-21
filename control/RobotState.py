class RobotState:

    def __init__(self, robot_id, path_segments):
        self.id = robot_id
        self.R = []
        self.PH = []
        self.status = "iddling"
        self.path_segments = path_segments
        self.current_t = 0.0

    def needs_resources(self):
        return [r for r in self.R if r not in self.PH]
        