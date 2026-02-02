import matplotlib.patches as patches
from control.RobotState import RobotState
from control.StagePassControl import StagePassControl
from control.RobotMotionControl import RobotMotionControl
import numpy as np


class AGV:

    def __init__(
            self, 
            agv_id: int, 
            marked_states: list[tuple[int, int]], 
            orientation: tuple[int, int], 
            radius: float,
            max_v: float,
            max_a: float,
            color: str, 
            path_color: str
        ):
        self.id = agv_id
        self.marked_states = marked_states
        self.orientation = orientation
        self.radius = radius
        self.color = color
        self.path_color = path_color
        self.t = 0.0

        self.path = []
        self.path_sectors = {}
        self.path_lengths = []

        self.state = RobotState(agv_id, radius, max_v, max_a)
        self.stage_pass = StagePassControl(self)
        self.motion_controller = RobotMotionControl(self)

        self.render = patches.Circle(self.marked_states[0], self.radius, color=self.color)

    def __str__(self):
        return f"AGV: marked_states: {self.marked_states}, radius: {self.radius}, color: {self.color}, path_color: {self.path_color}"
    
    def get_current_curve_sectors(self):
        return self.path_sectors.get(self.state.current_curve_idx, [])
    
    def init_path_lengths(self):
        self.path_lengths = []
        for curve_verts in self.path:
            length = self.calculate_bezier_length(curve_verts)
            self.path_lengths.append(length)
    
    def add_sector_to_curve(self, curve_idx, sector):
        if curve_idx not in self.path_sectors:
            self.path_sectors[curve_idx] = []
        self.path_sectors[curve_idx].append(sector)

    def update_position(self, new_t, curve_idx):
        self.t = new_t
        self.state.current_t = new_t
        self.state.current_curve_idx = curve_idx

    def _precompute_lengths(self):
        lengths = []
        for verts in self.path:
            lengths.append(self.calculate_bezier_length(verts))
        return lengths

    def calculate_bezier_length(self, verts, n=20):
        p0, p1, p2 = map(np.array, verts)
        pts = []
        for t in np.linspace(0, 1, n):
            pt = (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
            pts.append(pt)
        pts = np.array(pts)
        dist = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
        return np.sum(dist)
    
    def get_current_curve_length(self, idx):
        return self.path_lengths[idx] if idx < len(self.path_lengths) else 1.0

    def step(self, dt):
        v_target = self.stage_pass.get_setpoint()
        
        v_actual = self.motion_controller.compute_velocity(v_target, dt)
        
        L = self.get_current_curve_length(self.state.current_curve_idx)
        delta_t = (v_actual * dt) / L
        
        self.state.current_t += delta_t
        
        if self.state.current_t >= 1.0:
            self.state.current_t = 0.0
            self.state.current_curve_idx = (self.state.current_curve_idx + 1) % len(self.path)
