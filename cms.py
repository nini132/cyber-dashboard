#!/usr/bin/env python3
"""
COMPLETE SECURITY MONITORING SYSTEM - SINGLE FILE SOLUTION
Features:
- Auto-installs all dependencies
- RTSP/ONVIF Camera Support
- Splunk SIEM Integration
- Motion Detection with OpenCV
- Modern PyQt6 GUI with Dark Mode
- Complete error handling
- EDR/SOAR Integration (Conceptual)
"""

import os
import sys
import subprocess
import platform
import time
from typing import List, Dict, Optional
from enum import Enum

# ==================== AUTO-INSTALLER ====================
def install_packages():
    required = {
        'opencv-python': 'cv2',
        'numpy': 'numpy',
        'PyQt6': 'PyQt6',
        'requests': 'requests',
        'pyyaml': 'yaml',
        'Pillow': 'PIL',
        'onvif-zeep': 'onvif',  # Use this fork instead, it should work better with newer zeep versions
        'zeep': 'zeep',  # Add zeep (required by onvif)
    }

    for pkg, import_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            print(f"⚙️ Installing {pkg}...")
            if pkg == 'onvif-zeep':
                subprocess.check_call([sys.executable, "-m", "pip", "install", "onvif-zeep"])  # force install this and skip the older onvif
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

install_packages()

# ==================== CORE IMPORTS ====================
import cv2
import numpy as np
import requests
import yaml
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QComboBox, QSlider, QTabWidget,
    QGroupBox, QStatusBar, QFileDialog, QMessageBox, QLineEdit, QFormLayout,
    QListWidget, QListWidgetItem, QInputDialog
)
from PyQt6.QtGui import (
    QImage, QPixmap, QIcon, QPalette, QColor, QAction, QFont,
    QPainter, QTextCursor
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QPoint, QSize,
    QPropertyAnimation, QEasingCurve
)

from onvif import ONVIFCamera  # Add ONVIF imports, from onvif-zeep
from zeep.exceptions import Fault  # Add zeep Fault


# ==================== SYSTEM COMPONENTS ====================
class SecurityCamera:
    def __init__(self, config: dict):
        self.config = config
        self.cap = None  # Initialize to None
        self.cam = None  # ONVIFCamera instance
        self.prev_frame = None
        self.connect()  # Attempt to connect on initialization

    def connect(self):
        try:
            if self.config.get('onvif', False):
                self.cam = ONVIFCamera(
                    self.config['host'],
                    self.config['port'],
                    self.config['user'],
                    self.config['password']
                )
                self.cam.create_media_service()  # Needed for older versions of onvif

                self.ptz = self.cam.create_ptz_service()  # Create PTZ service
                # Get available PTZ configurations
                self.ptz_configurations = self.ptz.GetConfigurations()
                if self.ptz_configurations:  # Check if configurations are available
                    self.ptz_configuration_token = self.ptz_configurations[0].token  # Get the first configuration token. This might need adjustment based on your camera.
                else:
                    print("No PTZ configurations found for this camera.")
                    self.ptz = None  # Ensure self.ptz is set to None if unavailable

                self.profile = self.cam.media.GetProfiles()[0]  # get the first profile
                self.cap = cv2.VideoCapture(self.profile.MediaUri.Uri)

            else:
                self.cap = cv2.VideoCapture(self.config['url'])
        except Fault as e:
            print(f"ONVIF Fault: {e}")
            self.cap = cv2.VideoCapture(self.config['url'])  # Fallback to URL if ONVIF fails
        except Exception as e:
            print(f"ONVIF Connection Error: {e}")
            self.cap = None  # Indicate failure
            return  # Exit the function early

    def get_frame(self) -> Optional[np.ndarray]:
        if self.cap is None:  # Check if camera connection failed.
            print("Camera not connected.")
            return None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print(f"Error reading frame from {self.config['url']}")
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def detect_motion(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.prev_frame is None:
            self.prev_frame = gray
            return False

        diff = cv2.absdiff(self.prev_frame, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        self.prev_frame = gray
        return cv2.countNonZero(thresh) > self.config['motion_threshold']

    def move_ptz(self, pan: float, tilt: float, zoom: float):
        """Move the PTZ camera.  Pan, tilt, and zoom are floats between -1.0 and 1.0."""
        if self.cam and self.ptz:
            try:
                # Create ptz request object
                req = self.ptz.create_type('ContinuousMove')
                req.ProfileToken = self.profile.token

                req.Velocity = self.ptz.create_type('PTZVector')
                req.Velocity.PanTilt = self.ptz.create_type('Vector2D')
                req.Velocity.Zoom = self.ptz.create_type('Vector1D')

                req.Velocity.PanTilt.x = pan
                req.Velocity.PanTilt.y = tilt
                req.Velocity.Zoom.x = zoom

                self.ptz.ContinuousMove(req)
            except Fault as e:
                print(f"PTZ Move Error: {e}")
            except Exception as e:
                print(f"PTZ General Error: {e}")

    def stop_ptz(self):
        if self.cam and self.ptz:
            try:
                req = self.ptz.create_type('Stop')
                req.ProfileToken = self.profile.token
                req.PanTilt = True
                req.Zoom = True
                self.ptz.Stop(req)
            except Fault as e:
                print(f"PTZ Stop Error: {e}")
            except Exception as e:
                print(f"PTZ stop general error: {e}")

    def __del__(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()  # release the camera properly.


class SIEMClient:
    def __init__(self, config: dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config['token']}",
            "Content-Type": "application/json"
        })

    def fetch_alerts(self) -> List[Dict]:
        try:
            response = self.session.post(
                f"{self.config['api_url']}/services/search/jobs",
                data={"search": self.config['query']},
                timeout=5
            )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            json_response = response.json()
            return json_response.get('results', [])[:10]  # Return first 10 alerts
        except requests.exceptions.RequestException as e:  # Catch connection errors, timeout, etc.
            print(f"SIEM Error: {e}")
            return []
        except ValueError as e:  # Catch json decoding errors
            print(f"SIEM Error decoding JSON: {e}. Response Text: {response.text if 'response' in locals() else 'No response'}")
            return []


class EDRClient:  # Conceptual EDR client
    def __init__(self, config: dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config['token']}",
            "Content-Type": "application/json"
        })

    def send_alert(self, alert_data: dict):
        """Send alert to EDR system. (Conceptual)"""
        try:
            response = self.session.post(
                f"{self.config['api_url']}/alerts",
                json=alert_data,
                timeout=5
            )
            response.raise_for_status()
            print(f"EDR Alert Sent (Conceptual): {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"EDR Error: {e}")


class SOARClient:  # Conceptual SOAR client
    def __init__(self, config: dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config['token']}",
            "Content-Type": "application/json"
        })

    def send_alert(self, alert_data: dict):
        """Send alert to SOAR system. (Conceptual)"""
        try:
            response = self.session.post(
                f"{self.config['api_url']}/alerts",
                json=alert_data,
                timeout=5
            )
            response.raise_for_status()
            print(f"SOAR Alert Sent (Conceptual): {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"SOAR Error: {e}")


# ==================== GUI COMPONENTS ====================
class CameraThread(QThread):
    new_frame = pyqtSignal(np.ndarray)
    motion_detected = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.running = True

    def run(self):
        while self.running:
            try:
                frame = self.camera.get_frame()
                if frame is not None:
                    self.new_frame.emit(frame)
                    if self.camera.detect_motion(frame):
                        self.motion_detected.emit()
                else:
                    self.error_occurred.emit("Failed to retrieve frame from camera.")  # added error handling
                    time.sleep(5)
            except Exception as e:
                self.error_occurred.emit(str(e))
                time.sleep(5)

    def stop(self):
        self.running = False
        self.wait()


class NotificationManager:
    def __init__(self, parent):
        self.parent = parent
        self.notification = QLabel(parent)
        self.notification.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.notification.setStyleSheet("""
            background-color: rgba(231, 76, 60, 0.9);
            color: white;
            border-radius: 15px;
            padding: 15px;
            font-weight: bold;
            font-size: 14px;
            border: 2px solid #c0392b;
        """)
        self.notification.hide()
        self.notification.setFixedWidth(300)
        self.notification.setWordWrap(True)

        self.animation = QPropertyAnimation(self.notification, b"pos")
        self.animation.setEasingCurve(QEasingCurve.Type.OutBounce)
        self.animation.setDuration(800)

    def show(self, message: str, duration: int = 5000):
        self.notification.setText(message)
        self.notification.adjustSize()

        start_pos = QPoint(
            (self.parent.width() - self.notification.width()) // 2,
            -self.notification.height()
        )
        end_pos = QPoint(
            (self.parent.width() - self.notification.width()) // 2,
            20
        )

        self.notification.move(start_pos)
        self.notification.show()

        self.animation.setStartValue(b"pos")
        self.animation.setEndValue(end_pos)
        self.animation.start()

        QTimer.singleShot(duration, self.hide)

    def hide(self):
        self.animation.setDirection(QPropertyAnimation.Direction.Backward)
        self.animation.start()


class SecurityMonitorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Security Monitor Pro")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(800, 600)

        # Initialize camera configurations (replace with your actual configurations)
        self.camera_configs = [
            {"name": "Camera 1", "url": "rtsp://admin:password@192.168.1.10/stream", "motion_threshold": 5000, 'onvif': False},
            {"name": "Camera 2", "host": "192.168.1.20", "port": 80, "user": "admin", "password": "password", "motion_threshold": 5000, 'onvif': True},  # ONVIF Example
        ]
        self.config = {  # This is an ugly hack to share the siem config and other config across functions.
            "siem": {
                "api_url": "https://splunk.example.com:8089",
                "token": "your_api_token_here",
                "query": "search index=security earliest=-15m"
            },
            "edr": {  # Added EDR Config
                "api_url": "https://edr.example.com/api",
                "token": "your_edr_token_here"
            },
            "soar": {  # Added SOAR config
                "api_url": "https://soar.example.com/api",
                "token": "your_soar_token_here"
            }
        }

        self.cameras = {}  # Dictionary of camera objects.
        self.camera_threads = {}  # Dictionary of camera threads.

        # Initialize components
        self.init_ui()
        self.init_system()

        # Apply dark theme
        self.apply_dark_theme()

    def init_ui(self):
        """Initialize the user interface"""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main layout
        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        # Create menu bar
        self.init_menu_bar()

        # Create tabs
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # Add tabs
        self.create_monitor_tab()
        self.create_settings_tab()

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Notification system
        self.notifications = NotificationManager(self)

    def init_menu_bar(self):
        """Initialize the menu bar"""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def init_system(self):
        """Initialize the security system components"""
        self._init_cameras()
        self._init_siem_edr_soar()

    def _init_cameras(self):
        # Stop existing threads, and release existing camera objects.
        for camera_thread in self.camera_threads.values():
            camera_thread.stop()

        for camera in self.cameras.values():
            del camera  # Call destructor

        self.cameras = {}
        self.camera_threads = {}

        for config in self.camera_configs:
            try:
                camera = SecurityCamera(config)
                self.cameras[config['name']] = camera
                camera_thread = CameraThread(camera)
                self.camera_threads[config['name']] = camera_thread
                camera_thread.new_frame.connect(lambda frame, name=config['name']: self.update_video_frame(frame, name))
                camera_thread.motion_detected.connect(self.on_motion_detected)
                camera_thread.error_occurred.connect(self.on_camera_error)
                camera_thread.start()

            except Exception as e:
                self.notifications.show(f"Camera Error: {str(e)}", 5000)

    def _init_siem_edr_soar(self):
        """Initializes the SIEM, EDR, and SOAR clients."""
        # Initialize SIEM
        try:
            self.siem = SIEMClient(self.config['siem'])
        except Exception as e:
            self.notifications.show(f"SIEM Initialization Error: {str(e)}", 5000)

        # Initialize EDR Client
        try:
            self.edr = EDRClient(self.config['edr'])
        except Exception as e:
            self.notifications.show(f"EDR Initialization Error: {str(e)}", 5000)

        # Initialize SOAR Client
        try:
            self.soar = SOARClient(self.config['soar'])
        except Exception as e:
            self.notifications.show(f"SOAR Initialization Error: {str(e)}", 5000)

    def refresh_alerts(self):
        """Refresh SIEM alerts"""
        try:
            alerts = self.siem.fetch_alerts()
            self.alerts_display.clear()

            if not alerts:
                self.alerts_display.append("No alerts found")
                return

            for alert in alerts:
                self.alerts_display.append(
                    f"🔴 {alert.get('source', 'Unknown')}\n"
                    f"Severity: {alert.get('severity', 'N/A')}\n"
                    f"{alert.get('_raw', 'No details')}\n"
                    f"{'-'*50}"
                )
        except Exception as e:
            self.notifications.show(f"SIEM Error: {str(e)}", 5000)

    def send_alert_to_edr(self, alert_data: dict):
        """Send alert to EDR system. (Conceptual)"""
        try:
            self.edr.send_alert(alert_data)  # EDR Client call
        except Exception as e:
            self.notifications.show(f"EDR Send Error: {str(e)}", 5000)

    def send_alert_to_soar(self, alert_data: dict):
        """Send alert to SOAR system. (Conceptual)"""
        try:
            self.soar.send_alert(alert_data)  # SOAR Client Call
        except Exception as e:
            self.notifications.show(f"SOAR Send Error: {str(e)}", 5000)

    def on_motion_detected(self):
        """Handle motion detection events"""
        self.notifications.show("Motion detected!", 3000)
        self.refresh_alerts()

        # Enrich the Alert and Send to EDR/SOAR
        for camera_name in self.cameras:
            alert_data = {
                "camera_name": camera_name,
                "timestamp": time.time(),
                "motion_level": self.cameras[camera_name].config['motion_threshold'],  # Example Data
                "url": self.cameras[camera_name].config['url'],  # Example Data
                # Add more relevant data here
            }
            # Send to EDR/SOAR
            self.send_alert_to_edr(alert_data)
            self.send_alert_to_soar(alert_data)

    def create_monitor_tab(self):
        """Create the monitoring tab"""
        tab = QWidget()
        layout = QHBoxLayout()
        tab.setLayout(layout)

        # Create a layout for the camera feeds.  We will put each camera in a group box.
        camera_layout = QVBoxLayout()

        for config in self.camera_configs:
            camera_name = config['name']

            # Video panel
            video_panel = QGroupBox(camera_name)
            video_layout = QVBoxLayout()

            video_label = QLabel()
            video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            video_label.setStyleSheet("background-color: black;")
            # store the video label
            setattr(self, f"{camera_name}_video_label", video_label)  # Dynamically named attributes

            # Video controls
            controls_layout = QHBoxLayout()

            record_btn = QPushButton("⏺ Record")
            record_btn.setCheckable(True)

            snapshot_btn = QPushButton("📸 Snapshot")
            snapshot_btn.clicked.connect(lambda checked=False, name=camera_name: self.take_snapshot(name))  # Pass the name

            controls_layout.addWidget(record_btn)
            controls_layout.addWidget(snapshot_btn)

            # PTZ Controls
            ptz_layout = QHBoxLayout()

            pan_left_btn = QPushButton("◀ Pan Left")
            pan_right_btn = QPushButton("Pan Right ▶")
            tilt_up_btn = QPushButton("▲ Tilt Up")
            tilt_down_btn = QPushButton("Tilt Down ▼")
            zoom_in_btn = QPushButton("Zoom In +")
            zoom_out_btn = QPushButton("Zoom Out -")
            stop_ptz_btn = QPushButton("Stop PTZ")

            # Connect PTZ buttons to functions
            pan_left_btn.pressed.connect(lambda: self.start_ptz(camera_name, -0.5, 0, 0))
            pan_left_btn.released.connect(lambda: self.stop_ptz(camera_name))
            pan_right_btn.pressed.connect(lambda: self.start_ptz(camera_name, 0.5, 0, 0))
            pan_right_btn.released.connect(lambda: self.stop_ptz(camera_name))
            tilt_up_btn.pressed.connect(lambda: self.start_ptz(camera_name, 0, 0.5, 0))
            tilt_up_btn.released.connect(lambda: self.stop_ptz(camera_name))
            tilt_down_btn.pressed.connect(lambda: self.start_ptz(camera_name, 0, -0.5, 0))
            tilt_down_btn.released.connect(lambda: self.stop_ptz(camera_name))
            zoom_in_btn.pressed.connect(lambda: self.start_ptz(camera_name, 0, 0, 0.5))
            zoom_in_btn.released.connect(lambda: self.stop_ptz(camera_name))
            zoom_out_btn.pressed.connect(lambda: self.start_ptz(camera_name, 0, 0, -0.5))
            zoom_out_btn.released.connect(lambda: self.stop_ptz(camera_name))
            stop_ptz_btn.clicked.connect(lambda: self.stop_ptz(camera_name))

            ptz_layout.addWidget(pan_left_btn)
            ptz_layout.addWidget(pan_right_btn)
            ptz_layout.addWidget(tilt_up_btn)
            ptz_layout.addWidget(tilt_down_btn)
            ptz_layout.addWidget(zoom_in_btn)
            ptz_layout.addWidget(zoom_out_btn)
            ptz_layout.addWidget(stop_ptz_btn)

            video_layout.addWidget(video_label, 90)
            video_layout.addLayout(controls_layout, 5)
            video_layout.addLayout(ptz_layout, 5)
            video_panel.setLayout(video_layout)

            camera_layout.addWidget(video_panel)  # add to the overall camera layout

        # Alerts panel (30% width)
        alerts_panel = QGroupBox("Security Alerts")
        alerts_layout = QVBoxLayout()

        self.alerts_display = QTextEdit()
        self.alerts_display.setReadOnly(True)
        self.alerts_display.setStyleSheet("font-family: monospace;")

        self.refresh_btn = QPushButton("🔄 Refresh Alerts")
        self.refresh_btn.clicked.connect(self.refresh_alerts)

        alerts_layout.addWidget(self.alerts_display, 80)
        alerts_layout.addWidget(self.refresh_btn, 20)
        alerts_panel.setLayout(alerts_layout)

        layout.addLayout(camera_layout, 70)
        layout.addWidget(alerts_panel, 30)

        self.tabs.addTab(tab, "Monitoring")

    def update_video_frame(self, frame: np.ndarray, camera_name: str):
        """Update the video display with new frame"""
        video_label = getattr(self, f"{camera_name}_video_label")
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def take_snapshot(self, camera_name: str):
        """Save current video frame to file"""
        video_label = getattr(self, f"{camera_name}_video_label")
        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Save Snapshot - {camera_name}", "", "Images (*.png *.jpg)"
        )

        if file_path:
            pixmap = video_label.pixmap()
            if pixmap:
                pixmap.save(file_path)
                self.notifications.show(f"Snapshot saved to {file_path}", 3000)

    def start_ptz(self, camera_name: str, pan: float, tilt: float, zoom: float):
        if camera_name in self.cameras:
            camera = self.cameras[camera_name]
            camera.move_ptz(pan, tilt, zoom)

    def stop_ptz(self, camera_name: str):
        if camera_name in self.cameras:
            camera = self.cameras[camera_name]
            camera.stop_ptz()

    def create_settings_tab(self):
        """Create the settings tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Camera settings
        cam_group = QGroupBox("Camera Settings")
        cam_layout = QVBoxLayout()

        # Camera List
        self.camera_list_widget = QListWidget()
        for config in self.camera_configs:
            item = QListWidgetItem(config['name'])
            self.camera_list_widget.addItem(item)
        cam_layout.addWidget(self.camera_list_widget)

        # Buttons for adding and removing cameras
        button_layout = QHBoxLayout()
        add_button = QPushButton("Add Camera")
        add_button.clicked.connect(self.add_camera_config)

        remove_button = QPushButton("Remove Camera")
        remove_button.clicked.connect(self.remove_camera_config)

        button_layout.addWidget(add_button)
        button_layout.addWidget(remove_button)

        cam_layout.addLayout(button_layout)

        cam_group.setLayout(cam_layout)

        # SIEM settings
        siem_group = QGroupBox("SIEM Settings")
        siem_layout = QVBoxLayout()

        form_layout_siem = QFormLayout()

        self.siem_url_input = QLineEdit(self.config['siem']['api_url'])
        self.siem_token_input = QLineEdit()
        self.siem_token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.siem_token_input.setText(self.config['siem']['token'])  # Populate from config

        form_layout_siem.addRow("SIEM URL:", self.siem_url_input)
        form_layout_siem.addRow("API Token:", self.siem_token_input)

        siem_layout.addLayout(form_layout_siem)
        siem_group.setLayout(siem_layout)

        # EDR settings
        edr_group = QGroupBox("EDR Settings (Conceptual)")
        edr_layout = QFormLayout()

        self.edr_url_input = QLineEdit(self.config['edr']['api_url'])
        self.edr_token_input = QLineEdit()
        self.edr_token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.edr_token_input.setText(self.config['edr']['token'])

        edr_layout.addRow("EDR URL:", self.edr_url_input)
        edr_layout.addRow("API Token:", self.edr_token_input)
        edr_group.setLayout(edr_layout)

        # SOAR settings
        soar_group = QGroupBox("SOAR Settings (Conceptual)")
        soar_layout = QFormLayout()

        self.soar_url_input = QLineEdit(self.config['soar']['api_url'])
        self.soar_token_input = QLineEdit()
        self.soar_token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.soar_token_input.setText(self.config['soar']['token'])

        soar_layout.addRow("SOAR URL:", self.soar_url_input)
        soar_layout.addRow("API Token:", self.soar_token_input)
        soar_group.setLayout(soar_layout)

        # Save button
        save_btn = QPushButton("💾 Save Settings")
        save_btn.clicked.connect(self.save_settings)

        # Add to layout
        layout.addWidget(cam_group)
        layout.addWidget(siem_group)
        layout.addWidget(edr_group)
        layout.addWidget(soar_group)
        layout.addWidget(save_btn)
        layout.addStretch()

        self.tabs.addTab(tab, "Settings")

    def add_camera_config(self):
        """Add a new camera configuration"""
        camera_name, ok = QInputDialog.getText(self, "Add Camera", "Camera Name:")
        if ok and camera_name:
            # Basic default configuration
            new_config = {
                "name": camera_name,
                "url": "rtsp://admin:password@192.168.1.10/stream",
                "motion_threshold": 5000,
                "onvif": False  # Default to false
            }
            self.camera_configs.append(new_config)
            item = QListWidgetItem(camera_name)
            self.camera_list_widget.addItem(item)

    def remove_camera_config(self):
        """Remove the selected camera configuration"""
        selected_item = self.camera_list_widget.currentItem()
        if selected_item:
            camera_name = selected_item.text()
            for config in self.camera_configs:
                if config['name'] == camera_name:
                    self.camera_configs.remove(config)  # Remove from config.
                    break

            self.camera_list_widget.takeItem(self.camera_list_widget.row(selected_item))  # Remove from list widget

    def save_settings(self):
        """Save settings from UI"""
        self.config['siem']['api_url'] = self.siem_url_input.text()
        self.config['siem']['token'] = self.siem_token_input.text()

        self.config['edr']['api_url'] = self.edr_url_input.text()
        self.config['edr']['token'] = self.edr_token_input.text()

        self.config['soar']['api_url'] = self.soar_url_input.text()
        self.config['soar']['token'] = self.soar_token_input.text()

        self.notifications.show("Settings saved successfully!", 3000)

        # Reinitialize system with new settings
        self.init_system()

    def apply_dark_theme(self):
        """Apply a dark theme to the application"""
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Highlight, QColor(142, 45, 197))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        self.setPalette(palette)

    def show_about(self):
        """Show about dialog"""
        about_text = """
        <h2>Security Monitor Pro</h2>
        <p>Version: 1.0.0</p>
        <p>Developed by AI Assistant</p>
        <hr>
        <p>Comprehensive security monitoring solution with:</p>
        <ul>
            <li>RTSP Camera Support</li>
            <li>ONVIF Camera Support</li>
            <li>Splunk SIEM Integration</li>
            <li>Motion Detection</li>
            <li>Multiple Cameras</li>
            <li>Modern GUI Interface</li>
        </ul>
        """
        QMessageBox.about(self, "About Security Monitor", about_text)

    def closeEvent(self, event):
        """Clean up on application close"""
        for camera_thread in self.camera_threads.values():
            camera_thread.stop()
        event.accept()

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # Platform-specific setup
    if platform.system() == "Linux":
        os.environ["QT_QPA_PLATFORM"] = "xcb"

    # Create application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Set default font
    font = QFont()
    font.setFamily("Arial")
    font.setPointSize(10)
    app.setFont(font)

    # Create and show main window
    window = SecurityMonitorApp()
    window.show()

    # Run application
    sys.exit(app.exec())
