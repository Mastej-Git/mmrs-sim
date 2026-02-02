import numpy as np

class StagePassControl:

    def __init__(self, robot):
        self.robot = robot
        self.target_v = 0.0
        self.min_speed_ratio = 0.3

    def get_setpoint(self):
        status = self.robot.state.status
        max_v = self.robot.state.max_v
        
        if status == "iddling" or status == "finished":
            self.target_v = 0.0
        elif status == "running":
            self.target_v = self._compute_adaptive_speed(max_v)
            
        return self.target_v

    def _compute_adaptive_speed(self, max_v):
        current_curve_idx = self.robot.state.current_curve_idx
        current_t = self.robot.state.current_t

        if not self.robot.path or current_curve_idx >= len(self.robot.path):
            return max_v
        
        curve_length = self.robot.get_current_curve_length(current_curve_idx)
        max_a = self.robot.state.max_a
        min_speed = max_v * self.min_speed_ratio
        is_last_curve = (current_curve_idx >= len(self.robot.path) - 1)
        
        current_pos = current_t * curve_length
        remaining_dist = (1.0 - current_t) * curve_length
        
        end_speed = 0.0 if is_last_curve else min_speed
    
        max_speed_to_brake = np.sqrt(end_speed**2 + 2 * max_a * remaining_dist)
        speed_after_accel = np.sqrt(min_speed**2 + 2 * max_a * current_pos)
        target_speed = min(max_speed_to_brake, speed_after_accel, max_v)
        
        if not (is_last_curve and current_t > 0.9):
            target_speed = max(target_speed, min_speed)
        
        return target_speed
    
    def reset(self):
        self.target_v = 0.0
