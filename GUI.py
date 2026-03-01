from PyQt5.QtWidgets import ( 
    QMainWindow, 
    QTabWidget, 
    QWidget, 
    QVBoxLayout, 
    QFrame,
    QHBoxLayout,
    QSizePolicy,
    QLabel,
    QGridLayout,
    QGroupBox,
)
from PyQt5.QtCore import QTimer, Qt
from utils.StyleSheet import StyleSheet

from qt_widgets.ControlPanel import ControlPanel
from qt_widgets.Visualizer import Visualizer
from utils.YamlAGVLoader import YamlAGVLoader
import time
        

class GUI(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MMRS Simulator")
        self.setGeometry(100, 100, 1200, 800)

        self.simulation_start_time = None
        self.simulation_elapsed_time = 0.0
        self.is_simulation_running = False
        
        self.agv_times = {}

        self._update_timer = QTimer(self)
        self._update_timer.setInterval(40)
        self._update_timer.timeout.connect(self._on_update_tick)

        self._time_display_timer = QTimer(self)
        self._time_display_timer.setInterval(100)
        self._time_display_timer.timeout.connect(self._update_time_display)

        central_widget = QFrame()
        central_widget.setStyleSheet(StyleSheet.CentralWidget.value)

        layout = QHBoxLayout(central_widget)

        self.tabs = QTabWidget()
        self.tabs.tabBar().setExpanding(True)
        self.tabs.setStyleSheet(StyleSheet.Tab.value)

        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tabs.addTab(self.tab1, "Simulation")
        self.tabs.addTab(self.tab2, "Statistics")
        self.tabs.addTab(self.tab3, "")

        self.visualizer = Visualizer(self, width=5, height=4, dpi=100)
        self.yaml_agv_loader = YamlAGVLoader()

        self.agv_time_labels = {}
        self.system_time_label = None

        self.create_tabs_content()

        layout.addWidget(self.tabs, 1)
        # self.side_panel = self.create_control_panel()
        self.control_panel = ControlPanel()
        self.control_panel.assign_btn_connect_fns([
            self.on_run_clicked,
            self.on_pause_clicked,
            self.on_reset_clicked,
            self.on_show_paths_clicked,
            self.on_show_mpoints_clicked,
            self.on_show_lines_clicked,
            self.on_show_coll_sect_clicked,
            self.on_show_all_clicked,
            self.on_load_agv_clicked
        ])
        # layout.addWidget(self.side_panel)
        layout.addWidget(self.control_panel.upper_panel)
        self.setCentralWidget(central_widget)

    def create_tabs_content(self):
        layout1 = QVBoxLayout()
        layout1.addWidget(self.visualizer)
        self.tab1.setLayout(layout1)

        layout2 = QVBoxLayout()
        layout2.setContentsMargins(20, 20, 20, 20)
        layout2.setSpacing(15)

        system_group = QGroupBox("System Time")
        system_group.setStyleSheet(StyleSheet.QGroupBoxStatistics.value)
        system_layout = QVBoxLayout(system_group)
        
        self.system_time_label = QLabel("Total time: 0.00 s")
        # self.system_time_label.setStyleSheet(self._get_time_label_style(large=True))
        self.system_time_label.setStyleSheet(StyleSheet.TimeLabel(large=True))
        self.system_time_label.setAlignment(Qt.AlignCenter)
        system_layout.addWidget(self.system_time_label)
        
        self.system_status_label = QLabel("Status: Not started")
        self.system_status_label.setStyleSheet(StyleSheet.StatusLabel.value)
        self.system_status_label.setAlignment(Qt.AlignCenter)
        system_layout.addWidget(self.system_status_label)

        layout2.addWidget(system_group)

        self.robots_group = QGroupBox("Robot Times")
        self.robots_group.setStyleSheet(StyleSheet.QGroupBoxStatistics.value)
        self.robots_layout = QGridLayout(self.robots_group)
        self.robots_layout.setSpacing(10)
        
        headers = ["Robot", "Status", "Time"]
        for col, header in enumerate(headers):
            label = QLabel(header)
            label.setStyleSheet("font-weight: bold; font-size: 14px; color: #CCCCCC;")
            label.setAlignment(Qt.AlignCenter)
            self.robots_layout.addWidget(label, 0, col)

        self.no_robots_label = QLabel("No robots loaded")
        self.no_robots_label.setStyleSheet("color: #888888; font-style: italic;")
        self.no_robots_label.setAlignment(Qt.AlignCenter)
        self.robots_layout.addWidget(self.no_robots_label, 1, 0, 1, 3)

        layout2.addWidget(self.robots_group)
        layout2.addStretch(1)

        self.tab2.setLayout(layout2)

    def _get_time_label_style(self, large=False):
        size = "24px" if large else "14px"
        return f"""
            QLabel {{
                font-size: {size};
                font-weight: bold;
                color: #00FF00;
                padding: 10px;
            }}
        """

    def _init_robot_time_labels(self):
        for i in reversed(range(self.robots_layout.count())):
            item = self.robots_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if widget not in [self.robots_layout.itemAtPosition(0, c).widget() for c in range(3) if self.robots_layout.itemAtPosition(0, c)]:
                    widget.deleteLater()

        self.agv_time_labels = {}
        self.agv_times = {}

        agvs = self.visualizer.supervisor.agvs
        
        if not agvs:
            self.no_robots_label = QLabel("No robots loaded")
            self.no_robots_label.setStyleSheet("color: #888888; font-style: italic;")
            self.no_robots_label.setAlignment(Qt.AlignCenter)
            self.robots_layout.addWidget(self.no_robots_label, 1, 0, 1, 3)
            return

        for row, agv in enumerate(agvs, start=1):
            name_label = QLabel(f"AGV {agv.id}")
            name_label.setStyleSheet(f"color: {agv.path_color}; font-weight: bold; font-size: 14px;")
            name_label.setAlignment(Qt.AlignCenter)
            self.robots_layout.addWidget(name_label, row, 0)

            status_label = QLabel("Waiting")
            status_label.setStyleSheet("color: #AAAAAA; font-size: 14px;")
            status_label.setAlignment(Qt.AlignCenter)
            self.robots_layout.addWidget(status_label, row, 1)

            time_label = QLabel("0.00 s")
            time_label.setStyleSheet("color: #00FF00; font-size: 14px; font-weight: bold;")
            time_label.setAlignment(Qt.AlignCenter)
            self.robots_layout.addWidget(time_label, row, 2)

            self.agv_time_labels[agv.id] = {
                "name": name_label,
                "status": status_label,
                "time": time_label
            }

            self.agv_times[agv.id] = {
                "start": None,
                "end": None,
                "elapsed": 0.0,
                "finished": False
            }

    def _update_time_display(self):
        if not self.is_simulation_running:
            return

        current_time = time.time()
        
        elapsed = self.simulation_elapsed_time + (current_time - self.simulation_start_time)
        self.system_time_label.setText(f"Total time: {elapsed:.2f} s")

        for agv in self.visualizer.supervisor.agvs:
            agv_id = agv.id
            if agv_id not in self.agv_times:
                continue

            agv_data = self.agv_times[agv_id]
            labels = self.agv_time_labels.get(agv_id)
            
            if not labels:
                continue

            if agv.state.status == "finished" and not agv_data["finished"]:
                agv_data["finished"] = True
                agv_data["end"] = current_time
                agv_data["elapsed"] = agv_data["elapsed"] + (current_time - agv_data["start"]) if agv_data["start"] else 0.0
                labels["status"].setText("Finished")
                labels["status"].setStyleSheet("color: #00FF00; font-size: 14px;")
                labels["time"].setText(f"{agv_data['elapsed']:.2f} s")
                
            elif not agv_data["finished"]:
                if agv_data["start"]:
                    current_elapsed = agv_data["elapsed"] + (current_time - agv_data["start"])
                    labels["time"].setText(f"{current_elapsed:.2f} s")
                
                if agv.state.status == "running":
                    labels["status"].setText("Running")
                    labels["status"].setStyleSheet("color: #00AAFF; font-size: 14px;")
                elif agv.state.status == "iddling":
                    labels["status"].setText("Waiting")
                    labels["status"].setStyleSheet("color: #FFAA00; font-size: 14px;")

        all_finished = all(
            self.agv_times[agv.id]["finished"] 
            for agv in self.visualizer.supervisor.agvs 
            if agv.id in self.agv_times
        )
        
        if all_finished and self.visualizer.supervisor.agvs:
            self.system_status_label.setText("Status: All robots finished")
            self.system_status_label.setStyleSheet("color: #00FF00; font-size: 14px;")
            self._stop_timing()
            self.on_pause_clicked()
        else:
            self.system_status_label.setText("Status: Running")
            self.system_status_label.setStyleSheet("color: #00AAFF; font-size: 14px;")

    def _start_timing(self):
        current_time = time.time()
        self.simulation_start_time = current_time
        self.is_simulation_running = True
        
        for agv in self.visualizer.supervisor.agvs:
            if agv.id in self.agv_times and not self.agv_times[agv.id]["finished"]:
                if self.agv_times[agv.id]["start"] is None:
                    self.agv_times[agv.id]["start"] = current_time
                else:
                    self.agv_times[agv.id]["start"] = current_time

        self._time_display_timer.start()

    def _stop_timing(self):
        if not self.is_simulation_running:
            return

        current_time = time.time()
        
        self.simulation_elapsed_time += (current_time - self.simulation_start_time)
        
        for agv in self.visualizer.supervisor.agvs:
            if agv.id in self.agv_times and not self.agv_times[agv.id]["finished"]:
                if self.agv_times[agv.id]["start"]:
                    self.agv_times[agv.id]["elapsed"] += (current_time - self.agv_times[agv.id]["start"])
                    self.agv_times[agv.id]["start"] = None

        self.is_simulation_running = False
        self._time_display_timer.stop()
        
        self.system_status_label.setText("Status: Paused")
        self.system_status_label.setStyleSheet("color: #FFAA00; font-size: 14px;")

    def _reset_timing(self):
        self.simulation_start_time = None
        self.simulation_elapsed_time = 0.0
        self.is_simulation_running = False
        self._time_display_timer.stop()

        self.system_time_label.setText("Total time: 0.00 s")
        self.system_status_label.setText("Status: Not started")
        self.system_status_label.setStyleSheet(StyleSheet.StatusLabel.value)

        for agv_id, data in self.agv_times.items():
            data["start"] = None
            data["end"] = None
            data["elapsed"] = 0.0
            data["finished"] = False
            
            if agv_id in self.agv_time_labels:
                labels = self.agv_time_labels[agv_id]
                labels["status"].setText("Waiting")
                labels["status"].setStyleSheet("color: #AAAAAA; font-size: 14px;")
                labels["time"].setText("0.00 s")

    def on_run_clicked(self):
        self.control_panel.btn_run.setEnabled(False)
        self.control_panel.btn_pause.setEnabled(True)

        self.visualizer.timer.start(50)
        self.visualizer.simulation_f = True
        
        self._start_timing()

    def on_pause_clicked(self):
        self.control_panel.btn_run.setEnabled(True)
        self.control_panel.btn_pause.setEnabled(False)

        self.visualizer.timer.stop()
        self.visualizer.simulation_f = False
        
        self._stop_timing()

    def on_reset_clicked(self):
        self.control_panel.btn_run.setEnabled(True)
        self.control_panel.btn_pause.setEnabled(False)
        self.visualizer.reset_simulation()
        
        self._reset_timing()

    def on_show_paths_clicked(self):
        if self.control_panel.btn_show_paths.isChecked():
            self.control_panel.btn_show_paths.setText("Hide Paths")
            for i in range(self.visualizer.supervisor.get_agvs_number()):
                self.visualizer.draw_curve(i)
        else:
            self.control_panel.btn_show_paths.setText("Show Paths")
            self.visualizer.remove_curves()
        self.visualizer.draw()

    def on_show_mpoints_clicked(self):
        if self.control_panel.btn_show_points.isChecked():
            self.control_panel.btn_show_points.setText("Hide Points")
            for i in range(self.visualizer.supervisor.get_agvs_number()):
                self.visualizer.draw_middle_points(i)
        else:
            self.control_panel.btn_show_points.setText("Show Points")
            self.visualizer.remove_middle_points()
        self.visualizer.draw()

    def on_show_lines_clicked(self):
        if self.control_panel.btn_show_lines.isChecked():
            self.control_panel.btn_show_lines.setText("Hide Lines")
            for i in range(self.visualizer.supervisor.get_agvs_number()):
                self.visualizer.draw_add_lines(i)
        else:
            self.control_panel.btn_show_lines.setText("Show Lines")
            self.visualizer.remove_lines()
        self.visualizer.draw()

    def on_show_coll_sect_clicked(self):
        if self.control_panel.btn_det_col_sec.isChecked():
            self.control_panel.btn_det_col_sec.setText("Hide Coll Sectors")
            self.visualizer.draw_coll_sectors()
        else:
            self.control_panel.btn_det_col_sec.setText("Show Coll Sectors")
            self.visualizer.remove_coll_sectors()
        self.visualizer.draw()

    def on_show_all_clicked(self):
        if self.control_panel.btn_show_all.isChecked():
            self.control_panel.btn_show_paths.setCheckable(False)
            self.control_panel.btn_show_points.setCheckable(False)
            self.control_panel.btn_show_lines.setCheckable(False)
            self.control_panel.btn_det_col_sec.setCheckable(False)
            self.control_panel.btn_show_paths.setText("Hide Paths")
            self.control_panel.btn_show_points.setText("Hide Points")
            self.control_panel.btn_show_lines.setText("Hide Lines")
            self.control_panel.btn_det_col_sec.setText("Hide Coll Sectors")
            self.control_panel.btn_show_all.setText("Hide All")
            for i in range(self.visualizer.supervisor.get_agvs_number()):
                self.visualizer.draw_curve(i)
                self.visualizer.draw_middle_points(i)
                self.visualizer.draw_add_lines(i)
                self.visualizer.draw_coll_sectors()
        else:
            self.control_panel.btn_show_paths.setCheckable(True)
            self.control_panel.btn_show_points.setCheckable(True)
            self.control_panel.btn_show_lines.setCheckable(True)
            self.control_panel.btn_det_col_sec.setCheckable(True)
            self.control_panel.btn_show_paths.setText("Show Paths")
            self.control_panel.btn_show_points.setText("Show Points")
            self.control_panel.btn_show_lines.setText("Show Lines")
            self.control_panel.btn_det_col_sec.setText("Show Coll Sectors")
            self.control_panel.btn_show_all.setText("Show All")
            self.visualizer.remove_curves()
            self.visualizer.remove_middle_points()
            self.visualizer.remove_lines()
            self.visualizer.remove_coll_sectors()
        self.visualizer.draw()

    def on_load_agv_clicked(self):
        agvs = self.yaml_agv_loader.load_agvs_yaml()
        self.visualizer.supervisor.load_agvs(agvs)
        self.visualizer.load_agvs_t()

        self.visualizer.supervisor.trigger_path_creation()
        
        self.visualizer.supervisor.detec_col_sectors()
        self.visualizer.supervisor.finalize_agv_sectors()
        self.visualizer.supervisor.global_merge()
        self.visualizer.supervisor.get_all_control_points()

        for i in range(self.visualizer.supervisor.get_agvs_number()):
            self.visualizer.draw_bezier_curve(i)
        self.visualizer.draw()

        self._init_robot_time_labels()
        self._reset_timing()

    def _on_update_tick(self):
        for w in (self.path_creation_algorithm, self.single_bc):
            for method in ("update_plot", "redraw", "update", "repaint"):
                if hasattr(w, method):
                    try:
                        getattr(w, method)()
                    except Exception:
                        pass
                    break

