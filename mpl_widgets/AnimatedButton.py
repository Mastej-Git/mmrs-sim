from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import (QPropertyAnimation,
                          QRect,
                          QEasingCurve)

from utils.StyleSheet import StyleSheet

class AnimatedButton(QPushButton):

    def __init__(self, text, size_x=0, size_y=0, parent=None):
        super().__init__(text, parent)

        self.setStyleSheet(StyleSheet.QPushButton.value)
        if size_x != 0 and size_y != 0:
            self.setFixedSize(size_x, size_y)

        self.clicked.connect(self.animate_click)

    def animate_click(self) -> None:

        original_geometry = self.geometry()
        scaled_geometry = QRect(
            original_geometry.x() - 5, 
            original_geometry.y() - 5, 
            original_geometry.width() + 10, 
            original_geometry.height() + 10
        )

        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(100)
        self.animation.setStartValue(original_geometry)
        self.animation.setEndValue(scaled_geometry)
        self.animation.setEasingCurve(QEasingCurve.OutQuad)
        
        self.animation.finished.connect(self.animate_reverse)
        
        self.animation.start()

    def animate_reverse(self) -> None:
        original_geometry = self.geometry()
        normal_geometry = QRect(
            original_geometry.x() + 5, 
            original_geometry.y() + 5, 
            original_geometry.width() - 10, 
            original_geometry.height() - 10
        )
        
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(100)
        self.animation.setStartValue(original_geometry)
        self.animation.setEndValue(normal_geometry)
        self.animation.setEasingCurve(QEasingCurve.InQuad)
        
        self.animation.start()
        