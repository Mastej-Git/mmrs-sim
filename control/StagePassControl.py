class StagePassControl:

    def __init__(self, robot):
        self.robot = robot
        self.target_v = 0.0

    def get_setpoint(self):

        status = self.robot.state.status
        max_v = self.robot.state.max_v
        
        if status == "running":
            self.target_v = max_v
        elif status == "iddling" or status == "finished":
            self.target_v = 0.0
            
        return self.target_v