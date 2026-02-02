import numpy as np

class RobotMotionControl:

    def __init__(self, robot):
        self.robot = robot
        self.current_velocity = 0.0
        self.Q = np.diag([1.0, 1.0]) 
        self.R = np.array([[0.5]])

    def compute_velocity(self, target_speed, dt):
        curr_status = self.robot.state.status
        
        if curr_status == "iddling":
            return 0.0
        
        max_a = self.robot.state.max_a
        max_v = self.robot.state.max_v

        max_delta_v = max_a * dt

        velocity_diff = target_speed - self.current_velocity

        if abs(velocity_diff) <= max_delta_v:
            self.current_velocity = target_speed
        elif velocity_diff > 0:
            self.current_velocity += max_delta_v
        else:
            self.current_velocity -= max_delta_v

        self.current_velocity = np.clip(self.current_velocity, 0.0, max_v)
            
        return self.current_velocity
    
    def reset(self):
        self.current_velocity = 0.0