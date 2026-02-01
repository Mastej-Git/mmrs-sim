import matplotlib.patches as patches
from control.RobotState import RobotState


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

        self.state = RobotState(agv_id, radius, max_v, max_a)
        self.path_sectors = {}

        self.render = patches.Circle(self.marked_states[0], self.radius, color=self.color)

    def __str__(self):
        return f"AGV: marked_states: {self.marked_states}, radius: {self.radius}, color: {self.color}, path_color: {self.path_color}"
    
    def get_current_curve_sectors(self):
        return self.path_sectors.get(self.state.current_curve_idx, [])
    
    def add_sector_to_curve(self, curve_idx, sector):
        if curve_idx not in self.path_sectors:
            self.path_sectors[curve_idx] = []
        self.path_sectors[curve_idx].append(sector)

    def update_position(self, new_t, curve_idx):
        self.t = new_t
        self.state.current_t = new_t
        self.state.current_curve_idx = curve_idx
