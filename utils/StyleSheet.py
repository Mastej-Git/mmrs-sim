from enum import Enum

class StyleSheet(Enum):

    QPushButton = """
        QPushButton {
            background-color: #404040;
            color: #00ffff;
            border: 1px solid #404040;
            height: 50px;
        }

        QPushButton:hover {
            background-color: #2e2e2e;
            border: 1px solid #00ffff;
        }
    """
    
    QRadioButton = """
        QRadioButton {
            color: #00ffff;
            padding: 5px;
            background-color: #2d2d2d;
        }

        QRadioButton::indicator:checked {
            background-color: #00ffff;
            border: 2px solid #00ffff;
        }

        QRadioButton:hover {
            background-color: #2e2e2e;
            border: 1px solid #00ffff;
        }

        QRadioButton::indicator:hover {
            border: 2px solid #00ffff;
        }
    """
    
    QComboBox = """
        QComboBox {
            background-color: #404040;
            color: #00ffff;
            height: 40px;
            border: 1px solid #404040;
            padding: 5px;
            border-radius: 3px;
            combobox-popup: 0;
        }

        QComboBox:hover {
            background-color: #2e2e2e;
            border: 1px solid #00ffff;
        }

        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 15px;
            border-left-width: 1px;
            border-left-color: #00ffff;
            border-left-style: solid;
            border-top-right-radius: 3px;
            border-bottom-right-radius: 3px;
            background-color: #404040;
        }

        QComboBox::down-arrow {
            image: url(down-arrow.png);
            width: 10px;
            height: 10px;
        }

        QComboBox QAbstractItemView {
            background-color: #404040;
            color: #00ffff;
            selection-background-color: #2e2e2e;
            selection-color: #00ffff;
            border: 1px solid #00ffff;
        }
    """
    
    QLabel = """
        QLabel {
            color: #00ffff;
            padding: 5px;
            border-radius: 3px;
            height: 30px;
            background-color: #2d2d2d;
        }
    """
    
    InfoLabel = """
        QLabel {
            color: #00ffff;
            padding: 5px;
            border-radius: 3px;
            height: 30px;
            background-color: #2e2e2e;
        }
    """
    
    Tab = """
        QTabWidget::pane { 
            border: none; 
        }

        QTabBar::tab {
            background: #2e2e2e; 
            color: #b1b1b1; 
            width: 260px; 
            height: 40px;
        }

        QTabBar::tab:selected { 
            background: #404040; 
            color: #00ffff; 
            font-weight: bold;
        }
    """
    
    SubTab = """
        QTabBar::tab {
            background: #2e2e2e;
            color: #b1b1b1;
            width: 50px;
            height: 140px;
        }

        QTabBar::tab:selected {
            background: #404040;
            color: #00ffff;
            font-weight: bold;
        }
    """
    
    App = """
        QWidget {
            background-color: #2e2e2e;
            color: #b1b1b1;
        }

        QLabel {
            color: #b1b1b1;
        }
    """
    
    CentralWidget = """
        QFrame {
            border: 1px solid #2e2e2e;
            border-radius: 10px;
            background-color: #2e2e2e;
        }
    """
    
    QGroupBox = """
        QGroupBox {
            border: 1px solid #b1b1b1;
        }
    """

    StatusLabel = """
        QLabel {
            font-size: 14px;
            color: #AAAAAA;
            padding: 5px;
        }
    """

    QGroupBoxStatistics = """
        QGroupBox {
            font-size: 16px;
            font-weight: bold;
            color: #FFFFFF;
            border: 2px solid #444444;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 5px;
        }
    """

    TimeLabel = lambda large=False: f"""
        QLabel {{
            font-size: {"24px" if large else "14px"};
            font-weight: bold;
            color: #00FF00;
            padding: 10px;
        }}
    """