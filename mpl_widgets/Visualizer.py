from PyQt5.QtCore import QTimer, Qt
import matplotlib as mpl
mpl.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
from control.StageTransitionControl import StageTransitionControl
import time


class Visualizer(FigureCanvas):
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)

        self.simulation_f = False

        self.supervisor = StageTransitionControl()
        self.visual_agvs = []

        self.t = []
        self.path_idx = []

        # self.draw_square_grid(20)
        self.set_axis_limits(25)

        self._drawn_elements = {
            'curves': [],
            'points': [],
            'lines': [],
            'csectors': []
        }
        self.curve_list = []

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_position_forward)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def start_moving(self) -> None:
        self.timer.start(50)

    def load_agvs_t(self) -> None:
        self.t = list(0.0 for _ in range(len(self.supervisor.agvs)))
        self.path_idx = list(0 for _ in range(len(self.supervisor.agvs)))

    def bezier_point(self, t: float, verts: list[tuple[int, int]]) -> tuple[float, float]:
        return (
            (1 - t) ** 2 * np.array(verts[0])
            + 2 * (1 - t) * t * np.array(verts[1])
            + t ** 2 * np.array(verts[2])
        )

    def draw_square_grid(self, size: int = 10) -> None:
        for x in range(size + 1):
            self.ax.axhline(x, color="gray", linewidth=0.5)
            self.ax.axvline(x, color="gray", linewidth=0.5)

    def set_axis_limits(self, size: int) -> None:
        self.ax.set_xlim(0, size)
        self.ax.set_ylim(0, size)
        self.ax.set_aspect("equal")

    def draw_curve(self, i: int) -> None:
        for path in self.supervisor.agvs[i].path:
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            path_draw = Path(path, codes)

            patch = patches.PathPatch(path_draw, facecolor="none", lw=2, edgecolor=self.supervisor.agvs[i].path_color)
            self.ax.add_patch(patch)
            # self.curve_list.append(patch)
            self._drawn_elements['curves'].append(patch)

    def remove_curves(self) -> None:
        for curve in self._drawn_elements['curves']:
            curve.remove()
        self._drawn_elements['curves'].clear()

    def draw_add_lines(self, i: int) -> None:
        for positions in self.supervisor.agvs[i].path:
            x, y = zip(*positions)
            line, = self.ax.plot(x, y, "ro--")
            self._drawn_elements['lines'].append(line)

    def remove_lines(self) -> None:
        for line in self._drawn_elements['lines']:
            line.remove()
        self._drawn_elements['lines'].clear()

    def draw_marked_states(self) -> None:
        for agv in self.supervisor.agvs:
            for marked_state in agv.marked_states:
                point = patches.Circle(marked_state, 0.1, color=agv.path_color, zorder=4)
                self.ax.add_patch(point)

    def draw_middle_points(self, i: int) -> None:
        for p in self.supervisor.agvs[i].path:
            point = patches.Circle(p[1], 0.1, color="#EADA62", zorder=4)
            self.ax.add_patch(point)
            self._drawn_elements['points'].append(point)

    def remove_middle_points(self) -> None:
        for middle_point in self._drawn_elements['points']:
            middle_point.remove()
        self._drawn_elements['points'].clear()

    def draw_sector_on_curve(self, verts, t_l: float, t_u: float) -> None:
        if t_u <= t_l:
            return
        ts = np.linspace(max(0.0, t_l), min(1.0, t_u), 80)
        pts = np.array([self.bezier_point(t, verts) for t in ts])
        csector, = self.ax.plot(pts[:, 0], pts[:, 1], color="#FF4136", linewidth=6.0, alpha=0.6, solid_capstyle='round', zorder=5)
        self._drawn_elements['csectors'].append(csector)

    def draw_coll_sectors(self) -> None:
        self.remove_coll_sectors()
        for agv in self.supervisor.agvs:
            for curve_idx, sectors in agv.path_sectors.items():
                verts = agv.path[curve_idx]
                for sector in sectors:
                    self.draw_sector_on_curve(verts, sector.t_l, sector.t_u)
                    # print(sector)

    def remove_coll_sectors(self) -> None:
        for csector in self._drawn_elements['csectors']:
            csector.remove()
        self._drawn_elements['csectors'].clear()

    def draw_one_coll_sector(self) -> None:
        sec1, sec2 = self.supervisor.col_sectors[0]
        self.draw_sector_on_curve(sec1.addresses[0], sec1.t_l, sec1.t_u,)
        self.draw_sector_on_curve(sec2.addresses[1], sec2.t_l, sec2.t_u,)
        self.supervisor.col_sectors.pop(0)

    def draw_bezier_curve(self, i: int) -> None:

        self.draw_marked_states()
        agv = patches.Circle(
            self.supervisor.agvs[i].marked_states[0],
            self.supervisor.agvs[i].radius,
            color=self.supervisor.agvs[i].color,
            zorder=3
        )
        self.visual_agvs.append(agv)
        self.ax.add_patch(self.visual_agvs[i])

    def update_position_forward(self) -> None:
        dt = 0.05
        for i, agv in enumerate(self.supervisor.agvs):
            # agv.update_position(self.t[i], self.path_idx[i])
            self.supervisor.process_agv_step(agv)

            # if agv.state.status == "running":
            #     self.t[i] += 0.02

            # if self.t[i] >= 1.0:
            #     agv.update_position(1.0, self.path_idx[i])
            #     self.supervisor.process_agv_step(agv)
                
            #     self.t[i] = 0.0
            #     self.path_idx[i] = (self.path_idx[i] + 1) % len(agv.path)
                
            #     agv.update_position(0.0, self.path_idx[i])
            #     self.supervisor.process_agv_step(agv)

            agv.step(dt)
            
            self.t[i] = agv.state.current_t
            self.path_idx[i] = agv.state.current_curve_idx

            new_center = self.bezier_point(self.t[i], self.supervisor.agvs[i].path[self.path_idx[i]])
            self.visual_agvs[i].center = new_center

        # for res_id, res_obj in self.supervisor.ram.global_resources.items():
        #     if len(res_obj.priority_list) > 0:
        #         print(f"Zasób {res_id} zajęty przez: {res_obj.priority_list}")

        self.draw()

    def update_position_back(self) -> None:
        self.timer.stop()
        self.simulation_f = False

        for i, agv in enumerate(self.supervisor.agvs):
            agv.reset()
            
            self.t[i] = 0.0
            self.path_idx[i] = 0

            if agv.path:
                starting_point = self.bezier_point(0.0, agv.path[0])
                self.visual_agvs[i].center = starting_point
        
        for res_id, res_obj in self.supervisor.ram.global_resources.items():
            res_obj.priority_list = []
        
        self.draw()

    def reset_simulation(self) -> None:
        self.timer.stop()
        self.simulation_f = False
        
        for i, agv in enumerate(self.supervisor.agvs):
            agv.reset()
            
            self.t[i] = 0.0
            self.path_idx[i] = 0

            if agv.path:
                starting_point = self.bezier_point(0.0, agv.path[0])
                self.visual_agvs[i].center = starting_point
        
        for res_id, res_obj in self.supervisor.ram.global_resources.items():
            res_obj.priority_list = []
        
        self.draw()
            
    
    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Space:
            if self.simulation_f is False:
                self.timer.start(50)
                self.simulation_f = True
            else:
                self.timer.stop()
                self.simulation_f = False

        if event.key() == Qt.Key_Q:
            exit(0)

        elif event.key() == Qt.Key_Right:
            self.update_position_forward()

        elif event.key() == Qt.Key_Left:
            self.update_position_back()

        elif event.key() == Qt.Key_R:
            self.reset_simulation()

