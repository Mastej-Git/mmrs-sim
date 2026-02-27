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
from typing import Callable
from utils.StyleSheet import StyleSheet

from mpl_widgets.AnimatedButton import AnimatedButton


class ControlPanel():

    def __init__(self):
        panel = QFrame()
        panel.setObjectName("controlPanel")
        panel.setFixedWidth(250)
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
        self.btn_show_lines = AnimatedButton("Show Lines")
        self.btn_det_col_sec = AnimatedButton("Show Coll Sectors")
        self.btn_show_all = AnimatedButton("Show All")

        for b in (self.btn_show_paths, self.btn_show_points, self.btn_show_lines, self.btn_det_col_sec, self.btn_show_all):
            b.setCheckable(True)
            b.setChecked(False)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            show_layout.addWidget(b)

        self.btn_show_paths.clicked.connect(self.on_show_paths_clicked)
        self.btn_show_points.clicked.connect(self.on_show_mpoints_clicked)
        self.btn_show_lines.clicked.connect(self.on_show_lines_clicked)
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

    def _create_upper_panel(self) -> QFrame:
        pass

    def _create_middle_panel(self) -> QFrame:
        pass

    def _create_lower_panel(self) -> QFrame:
        pass

    def assign_btn_connect_fns(self, fnc_list: list[Callable[..., None]]) -> None:
        pass

    def _assign_upper_panel_btn_connect_fnc(self, fnc_list: list[Callable[..., None]]) -> None:
        pass

    def _assign_middle_panel_btn_connect_fnc(self, fnc_list: list[Callable[..., None]]) -> None:
        pass

    def _assign_lower_panel_btn_connect_fnc(self, fnc_list: list[Callable[..., None]]) -> None:
        pass
