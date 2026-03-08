from PyQt5.QtWidgets import ( 
    QVBoxLayout, 
    QFrame,
    QSizePolicy,
    QLabel,
)
from typing import Callable
from utils.StyleSheet import StyleSheet

from qt_widgets.AnimatedButton import AnimatedButton


class ControlPanel():

    def __init__(self):

        self._create_upper_panel()
        self._create_middle_panel()
        self._create_lower_panel()

    def _create_upper_panel(self) -> QFrame:
        self.upper_panel = QFrame()
        self.upper_panel.setObjectName("controlPanel")
        self.upper_panel.setFixedWidth(250)
        self.upper_panel.setStyleSheet(StyleSheet.CentralWidget.value)

        self.vbox = QVBoxLayout(self.upper_panel)
        self.vbox.setContentsMargins(10, 10, 10, 10)
        self.vbox.setSpacing(8)

        title = QLabel("Controls")
        title.setStyleSheet(StyleSheet.InfoLabel.value)

        self.vbox.addWidget(title)

        self._create_upper_panel_buttons()

    def _create_middle_panel(self) -> QFrame:
        show_title = QLabel("Show elements")
        show_title.setStyleSheet(StyleSheet.InfoLabel.value)

        self.show_frame = QFrame()
        self.show_layout = QVBoxLayout(self.show_frame)
        self.show_layout.setContentsMargins(0, 0, 0, 0)
        self.show_layout.setSpacing(6)

        self.vbox.addWidget(show_title)
        self.vbox.addWidget(self.show_frame)

        self.vbox.addStretch(1)

        self._create_middle_panel_buttons()

    def _create_lower_panel(self) -> QFrame:
        load_title = QLabel("Load Configuration")
        load_title.setStyleSheet(StyleSheet.InfoLabel.value)
        self.bottom_frame = QFrame()
        self.bottom_layout = QVBoxLayout(self.bottom_frame)
        self.bottom_layout.setContentsMargins(0, 0, 0, 0)
        self.bottom_layout.setSpacing(6)

        self.vbox.addWidget(load_title)
        self.vbox.addWidget(self.bottom_frame)

        self._create_lower_panel_buttons()

    def _create_upper_panel_buttons(self) -> None:
        self.btn_run = AnimatedButton("Run")
        self.btn_pause = AnimatedButton("Pause")
        self.btn_reset = AnimatedButton("Reset")

        for b in (self.btn_run, self.btn_pause, self.btn_reset):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.vbox.addWidget(b)

        self.btn_pause.setEnabled(False)
        
    def _create_middle_panel_buttons(self) -> None:
        self.btn_show_paths = AnimatedButton("Show Paths")
        self.btn_show_points = AnimatedButton("Show Mid Points")
        self.btn_show_lines = AnimatedButton("Show Lines")
        self.btn_det_col_sec = AnimatedButton("Show Coll Sectors")
        self.btn_show_all = AnimatedButton("Show All")

        for b in (self.btn_show_paths, self.btn_show_points, self.btn_show_lines, self.btn_det_col_sec, self.btn_show_all):
            b.setCheckable(True)
            b.setChecked(False)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.show_layout.addWidget(b)

    def _create_lower_panel_buttons(self) -> None:
        self.btn_load_agv = AnimatedButton("Load AGVs")
        self.btn_load_map = AnimatedButton("Load Map")

        for b in (self.btn_load_agv, self.btn_load_map):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.bottom_layout.addWidget(b)

    def assign_btn_connect_fns(self, fnc_list: list[Callable[..., None]]) -> None:
        self._assign_upper_panel_btn_connect_fnc(fnc_list[0:3])
        self._assign_middle_panel_btn_connect_fnc(fnc_list[3:8])
        self._assign_lower_panel_btn_connect_fnc(fnc_list[8:])

    def _assign_upper_panel_btn_connect_fnc(self, fnc_list: list[Callable[..., None]]) -> None:
        self.btn_run.clicked.connect(fnc_list[0])
        self.btn_pause.clicked.connect(fnc_list[1])
        self.btn_reset.clicked.connect(fnc_list[2])

    def _assign_middle_panel_btn_connect_fnc(self, fnc_list: list[Callable[..., None]]) -> None:
        self.btn_show_paths.clicked.connect(fnc_list[0])
        self.btn_show_points.clicked.connect(fnc_list[1])
        self.btn_show_lines.clicked.connect(fnc_list[2])
        self.btn_det_col_sec.clicked.connect(fnc_list[3])
        self.btn_show_all.clicked.connect(fnc_list[4])

    def _assign_lower_panel_btn_connect_fnc(self, fnc_list: list[Callable[..., None]]) -> None:
        self.btn_load_agv.clicked.connect(fnc_list[0])
        self.btn_load_map.clicked.connect(fnc_list[1])
