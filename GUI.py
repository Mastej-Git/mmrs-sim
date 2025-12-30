from PyQt5.QtWidgets import ( 
    QMainWindow, 
    QTabWidget, 
    QWidget, 
    QVBoxLayout, 
    QFrame,
    QHBoxLayout,
    QSizePolicy,
    QLabel,
)
from PyQt5.QtCore import QTimer
from utils.StyleSheet import StyleSheet

from mpl_widgets.AnimatedButton import AnimatedButton
from mpl_widgets.Visualizer import Visualizer
from utils.YamlAGVLoader import YamlAGVLoader
        

class GUI(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MMRS Simulator")
        self.setGeometry(100, 100, 1200, 800)
        # start application in full screen
        # note: this will make the main window fullscreen when constructed
        # call showFullScreen() here so the app opens fullscreen instead of fixed 800x600
        # self.showFullScreen()

        # timer used for periodic updates when "Run" is active
        self._update_timer = QTimer(self)
        self._update_timer.setInterval(40)  # ~25 FPS, adjust as needed
        self._update_timer.timeout.connect(self._on_update_tick)

        central_widget = QFrame()
        central_widget.setStyleSheet(StyleSheet.CentralWidget.value)

        layout = QHBoxLayout(central_widget)

        self.tabs = QTabWidget()
        self.tabs.tabBar().setExpanding(True)
        self.tabs.setStyleSheet(StyleSheet.Tab.value)

        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tabs.addTab(self.tab1, "Tab 1")
        self.tabs.addTab(self.tab2, "Tab 2")
        self.tabs.addTab(self.tab3, "Tab 3")

        self.visualizer = Visualizer(self, width=5, height=4, dpi=100)
        self.yaml_agv_loader = YamlAGVLoader()

        self.create_tabs_content()

        layout.addWidget(self.tabs, 1)  # stretch factor 1 -> takes remaining space
        self.side_panel = self.create_control_panel()
        layout.addWidget(self.side_panel)
        self.setCentralWidget(central_widget)

    def create_control_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("controlPanel")
        panel.setFixedWidth(250)  # keep a constant width for the side panel
        # use existing central/app theme so the panel matches the rest of the UI
        # CentralWidget provides the same background as the main area
        panel.setStyleSheet(StyleSheet.CentralWidget.value)

        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(10, 10, 10, 10)
        vbox.setSpacing(8)

        title = QLabel("Controls")
        title.setStyleSheet(StyleSheet.InfoLabel.value)

        vbox.addWidget(title)

        self.btn_run = AnimatedButton("Run")
        self.btn_pause = AnimatedButton("Pause")
        self.btn_reset = AnimatedButton("Reset")

        for b in (self.btn_run, self.btn_pause, self.btn_reset):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            vbox.addWidget(b)

        self.btn_run.clicked.connect(self.on_run_clicked)
        self.btn_pause.clicked.connect(self.on_pause_clicked)
        self.btn_reset.clicked.connect(self.on_reset_clicked)

        self.btn_pause.setEnabled(False)

        show_title = QLabel("Show elements")
        show_title.setStyleSheet(StyleSheet.InfoLabel.value)

        show_frame = QFrame()
        show_layout = QVBoxLayout(show_frame)
        show_layout.setContentsMargins(0, 0, 0, 0)
        show_layout.setSpacing(6)

        self.btn_show_paths = AnimatedButton("Show Paths")
        self.btn_show_points = AnimatedButton("Show Mid Points")
        self.btn_show_lines = AnimatedButton("Show Add lines")
        self.btn_show_all = AnimatedButton("Show All")
        self.btn_det_col_sec = AnimatedButton("Define Coll Sectors")

        for b in (self.btn_show_paths, self.btn_show_points, self.btn_show_lines, self.btn_show_all, self.btn_det_col_sec):
            b.setCheckable(True)
            b.setChecked(True)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            show_layout.addWidget(b)

        self.btn_show_paths.clicked.connect(self.on_toggle_show_paths)
        self.btn_show_points.clicked.connect(self.on_toggle_show_points)
        self.btn_show_lines.clicked.connect(self.on_toggle_show_lines)
        self.btn_show_all.clicked.connect(self.on_show_all_clicked)
        self.btn_det_col_sec.clicked.connect(self.on_det_col_sec_clicked)


        vbox.addWidget(show_title)
        vbox.addWidget(show_frame)

        vbox.addStretch(1)

        load_title = QLabel("Load Configuration")
        load_title.setStyleSheet(StyleSheet.InfoLabel.value)
        bottom_frame = QFrame()
        bottom_layout = QVBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(6)

        self.btn_load_agv = AnimatedButton("Load AGVs")
        self.btn_load_map = AnimatedButton("Load Map")

        for b in (self.btn_load_agv, self.btn_load_map):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            bottom_layout.addWidget(b)

        self.btn_load_agv.clicked.connect(self.on_load_agv_clicked)

        vbox.addWidget(load_title)
        vbox.addWidget(bottom_frame)

        return panel

    def create_tabs_content(self):
        sub_tab_widget = QTabWidget()
        sub_tab_widget.setTabPosition(QTabWidget.West)
        sub_tab_widget.setStyleSheet(StyleSheet.SubTab.value)

        layout1 = QVBoxLayout()
        layout1.addWidget(self.visualizer)
        self.tab1.setLayout(layout1)

        layout2 = QVBoxLayout()
        self.tab2.setLayout(layout2)

    def on_run_clicked(self):
        self.btn_run.setEnabled(False)
        self.btn_pause.setEnabled(True)

        self.visualizer.timer.start(50)
        self.visualizer.simulation_f = True

    def on_pause_clicked(self):
        self.btn_run.setEnabled(True)
        self.btn_pause.setEnabled(False)

        self.visualizer.timer.stop()
        self.visualizer.simulation_f = False

    def on_reset_clicked(self):
        self.visualizer.reset_simulation()

    def on_show_all_clicked(self):
        for i in range(self.visualizer.supervisor.get_agvs_number()):
            self.visualizer.draw_curve(i)
            self.visualizer.draw_middle_points(i)
            self.visualizer.draw_add_lines(i)
        self.visualizer.draw()

    def on_det_col_sec_clicked(self):
        self.visualizer.draw_coll_sectors()
        # self.visualizer.draw_one_coll_sector()
        self.visualizer.draw()

    def on_toggle_show_paths(self):
        for i in range(self.visualizer.supervisor.get_agvs_number()):
            self.visualizer.draw_curve(i)
        self.visualizer.draw()

    def on_toggle_show_points(self):
        for i in range(self.visualizer.supervisor.get_agvs_number()):
            self.visualizer.draw_middle_points(i)
        self.visualizer.draw()

    def on_toggle_show_lines(self):
        for i in range(self.visualizer.supervisor.get_agvs_number()):
            self.visualizer.draw_add_lines(i)
        self.visualizer.draw()

    def on_load_agv_clicked(self):
        agvs = self.yaml_agv_loader.load_agvs_yaml()
        self.visualizer.supervisor.load_agvs(agvs)
        self.visualizer.load_agvs_t()
        self.visualizer.supervisor.trigger_path_creation()
        self.visualizer.supervisor.detec_col_sectors()
        for i in range(self.visualizer.supervisor.get_agvs_number()):
            self.visualizer.draw_bezier_curve(i)
        self.visualizer.draw()

    def _on_update_tick(self):
        for w in (self.path_creation_algorithm, self.single_bc):
            for method in ("update_plot", "redraw", "update", "repaint"):
                if hasattr(w, method):
                    try:
                        getattr(w, method)()
                    except Exception:
                        pass
                    break

