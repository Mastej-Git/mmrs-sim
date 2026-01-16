import matplotlib.patches as patches


class AGV:

    def __init__(self, marked_states: list[tuple[int, int]], orientation: tuple[int, int],radius: float, color: str, path_color: str):
        self.marked_states = marked_states
        self.orientation = orientation
        self.radius = radius
        self.color = color
        self.path_color = path_color
        self.t = 0.0
        self.path = []

        self.render = patches.Circle(self.marked_states[0], self.radius, color=self.color)

    def __str__(self):
        return f"AGV: marked_states: {self.marked_states}, radius: {self.radius}, color: {self.color}, path_color: {self.path_color}"
