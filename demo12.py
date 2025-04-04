"""
COMPLETE SECURITY MONITORING SYSTEM - SINGLE FILE SOLUTION
Features:
- Auto-installs all dependencies
- RTSP/ONVIF Camera Support (including PTZ)
- Splunk SIEM Integration
- Motion Detection with OpenCV
- Modern PyQt6 GUI with Dark Mode
- Dynamic Camera Configuration (Add/Remove)
- Visual Map View for Camera Placement
- Robust error handling and user feedback
"""

import os
import sys
import subprocess
import platform
import time
import datetime
import json # Needed for SIEM response parsing
from typing import List, Dict, Optional, Tuple, Any, Union # Added Union
from enum import Enum
import logging
import asyncio # For potential future ONVIF async operations if needed

# ==================== LOGGING SETUP ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get a logger instance for the module

# ==================== AUTO-INSTALLER ====================
def install_packages():
    """Checks and installs required Python packages using pip."""
    required = {
        'opencv-python': 'cv2',
        'numpy': 'numpy',
        'PyQt6': 'PyQt6',
        'requests': 'requests',
        'pyyaml': 'yaml',
        'Pillow': 'PIL',
        'onvif-zeep': 'onvif',
        'zeep': 'zeep',
    }
    installed_something = False
    for pkg, import_name in required.items():
        try:
            if '.' in import_name:
                base_module = import_name.split('.')[0]
                __import__(base_module)
            else:
                __import__(import_name)
        except ImportError:
            logger.info(f"⚙️ Package '{pkg}' not found. Installing...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "--upgrade", pkg, "--disable-pip-version-check", "--no-cache-dir"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE
                )
                logger.info(f"✅ Successfully installed {pkg}.")
                installed_something = True
            except subprocess.CalledProcessError as e:
                error_output = e.stderr.decode(errors='ignore') if e.stderr else "No error output"
                logger.error(f"❌ Failed to install {pkg}. Pip Error: {error_output[:500]}...")
                print(f"❌ Error installing {pkg}. Please install it manually using: "
                      f"'{sys.executable} -m pip install {pkg}' and restart the application.")
                sys.exit(1)
            except Exception as e:
                logger.error(f"❌ An unexpected error occurred during installation of {pkg}. Error: {e}")
                sys.exit(1)

# Run installer immediately
install_packages()


# ==================== CORE IMPORTS ====================
try:
    import cv2
    import numpy as np
    import requests
    import yaml
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QPushButton, QTextEdit, QComboBox, QSlider, QTabWidget, QToolBar,
        QGroupBox, QStatusBar, QFileDialog, QMessageBox, QLineEdit, QFormLayout,
        QListWidget, QListWidgetItem, QInputDialog, QDialog, QDialogButtonBox,
        QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem,
        QGraphicsDropShadowEffect, QCheckBox, QScrollArea
    )
    from PyQt6.QtGui import (
        QImage, QPixmap, QIcon, QPalette, QColor, QAction, QFont, QActionGroup,
        QPainter, QTextCursor, QPen, QBrush, QTransform
    )
    from PyQt6.QtCore import (
        Qt, QTimer, QThread, pyqtSignal, QPoint, QSize, QRectF, QPointF,
        QPropertyAnimation, QEasingCurve, QVariantAnimation, QRect, pyqtSlot, QMetaObject, Q_ARG
    )
    try:
        from onvif import ONVIFCamera
        from zeep.exceptions import Fault, TransportError, XMLSyntaxError
    except ImportError:
        ONVIFCamera = None
        Fault = TransportError = XMLSyntaxError = Exception
        logger.warning("onvif-zeep library not found or failed to import. ONVIF functionality will be disabled.")

    from requests.exceptions import ConnectionError as RequestsConnectionError, Timeout, RequestException
    from PIL import Image # Pillow import
except ImportError as e:
    logger.critical(f"❌ Critical import failed after installation attempt: {e}. Please ensure all dependencies were installed correctly.", exc_info=True)
    if QApplication.instance():
         QMessageBox.critical(None, "Import Error", f"Failed to import a required library: {e}\n\nPlease check installation and restart.")
    else:
         print(f"❌ Failed to import a required library: {e}. Exiting.", file=sys.stderr)
    sys.exit(1)


# ==================== SYSTEM COMPONENTS ====================

class SecurityCamera:
    """Handles connection, frame retrieval, motion detection, and PTZ for a single camera."""
    def __init__(self, config: dict):
        self.config = config
        self.name = config.get('name', f'UnnamedCamera_{int(time.time())}')
        self.url = config.get('url')
        self.is_onvif = config.get('onvif', False)
        self.host = config.get('host')
        self.port = config.get('port')
        self.user = config.get('user')
        self.password = config.get('password')
        self.motion_threshold = config.get('motion_threshold', 500)

        self.cap: Optional[cv2.VideoCapture] = None
        self.onvif_cam: Optional[ONVIFCamera] = None
        self.ptz = None
        self.media_profile = None
        self.ptz_configuration_token: Optional[str] = None
        self.prev_frame_gray: Optional[np.ndarray] = None

        self.is_connected: bool = False
        self.is_connecting: bool = False
        self.last_connection_attempt_time: float = 0
        self.connection_retry_delay: int = 15
        self.last_error: Optional[str] = None

        logger.debug(f"Initializing camera object: {self.name} (ONVIF: {self.is_onvif})")

    def _set_error(self, message: str, log_level=logging.ERROR):
        """Sets the last error message and updates connection state."""
        if log_level == logging.ERROR: logger.error(f"Camera Error ({self.name}): {message}")
        elif log_level == logging.WARNING: logger.warning(f"Camera Warning ({self.name}): {message}")
        else: logger.info(f"Camera Info ({self.name}): {message}")

        self.last_error = message
        if log_level >= logging.ERROR:
            if self.is_connected:
                logger.info(f"Camera '{self.name}' state changed to DISCONNECTED due to error.")
            self.is_connected = False
            self.release_capture()
            self.onvif_cam = None
            self.ptz = None
            self.media_profile = None

    def release_capture(self):
        """Releases the OpenCV VideoCapture object."""
        if self.cap is not None:
            logger.debug(f"Releasing video capture for {self.name}")
            try:
                self.cap.release()
            except Exception as e:
                logger.error(f"Exception releasing capture for {self.name}: {e}")
            self.cap = None

    # ***** CORRECTED connect METHOD *****
    def connect(self) -> bool:
        """Attempts to connect to the camera (ONVIF or RTSP URL). Returns True on success."""
        if self.is_connecting:
            logger.debug(f"Connection attempt for {self.name} skipped: Already connecting.")
            return False
        if self.is_connected:
             logger.debug(f"Connection attempt for {self.name} skipped: Already connected.")
             return True # Already connected

        current_time = time.time()
        if current_time - self.last_connection_attempt_time < self.connection_retry_delay:
             # logger.debug(f"Connection attempt for {self.name} skipped: Retry delay not elapsed.")
             return False # Still in retry delay

        self.is_connecting = True
        self.last_connection_attempt_time = current_time
        logger.info(f"Attempting connection to camera: {self.name}")
        self.last_error = None # Clear previous error on new attempt

        # Ensure previous resources are released before attempting connection
        self.release_capture()
        self.onvif_cam = None
        self.ptz = None

        stream_uri = self.url # Use configured URL as default/fallback

        try:
            # --- ONVIF Connection Logic ---
            if self.is_onvif:
                if not ONVIFCamera: # Check if ONVIF library failed to import
                    self._set_error("ONVIF connection failed: 'onvif-zeep' library not available.")
                    self.is_connecting = False
                    return False
                if not self.host or not self.port or self.user is None or self.password is None:
                     self._set_error("ONVIF connection failed: Host, Port, User, or Password missing in config.")
                     self.is_connecting = False
                     return False

                logger.info(f"Connecting to {self.name} via ONVIF: {self.host}:{self.port}")
                # Determine WSDL path robustly
                try:
                    import onvif
                    wsdl_dir = os.path.join(os.path.dirname(onvif.__file__), 'wsdl')
                    if not os.path.exists(wsdl_dir):
                         wsdl_dir_alt = os.path.join(os.path.dirname(os.path.dirname(onvif.__file__)), 'onvif_wsdl')
                         if os.path.exists(wsdl_dir_alt): wsdl_dir = wsdl_dir_alt
                         else:
                             wsdl_dir_script = os.path.join(os.path.dirname(__file__), 'wsdl')
                             if os.path.exists(wsdl_dir_script): wsdl_dir = wsdl_dir_script
                             else: raise FileNotFoundError("WSDL directory not found")
                    logger.debug(f"Using ONVIF WSDL directory: {wsdl_dir}")
                except Exception as e:
                    self._set_error(f"ONVIF WSDL files lookup failed: {e}. Cannot establish ONVIF connection.")
                    self.is_connecting = False
                    return False

                try:
                    self.onvif_cam = ONVIFCamera(
                        self.host, self.port, self.user, self.password,
                        wsdl_dir=wsdl_dir,
                        transport_timeout=10 # seconds
                    )
                    # ***** CORRECTION: Removed await/run_in_executor *****
                    logger.debug(f"Attempting to fetch device info for {self.name}...")
                    # Make the call directly (synchronously)
                    device_info = self.onvif_cam.devicemgmt.GetDeviceInformation()
                    logger.info(f"ONVIF device connected: {device_info.Manufacturer} {device_info.Model} ({device_info.FirmwareVersion})")

                    # --- Fetch Media Profiles and Stream URI ---
                    media_service = self.onvif_cam.create_media_service()
                    # ***** CORRECTION: Removed await/run_in_executor *****
                    profiles = media_service.GetProfiles()
                    if not profiles:
                        self._set_error("No media profiles found via ONVIF.")
                        self.is_connecting = False; return False

                    video_profiles = [p for p in profiles if hasattr(p, 'VideoEncoderConfiguration') and p.VideoEncoderConfiguration]
                    self.media_profile = video_profiles[0] if video_profiles else profiles[0]
                    logger.info(f"Using ONVIF media profile: {self.media_profile.Name} (Token: {self.media_profile.token})")

                    req = media_service.create_type('GetStreamUri')
                    req.ProfileToken = self.media_profile.token
                    obtained_uri = None
                    # Try TCP first
                    try:
                         req.StreamSetup = {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'TCP'}}
                         # ***** CORRECTION: Removed await/run_in_executor *****
                         stream_info = media_service.GetStreamUri(req)
                         obtained_uri = stream_info.Uri
                         logger.info(f"Got ONVIF stream URI (TCP): {obtained_uri}")
                    except (Fault, TransportError, TimeoutError, ConnectionRefusedError) as e_tcp:
                         logger.warning(f"Failed to get TCP stream URI for {self.name}: {e_tcp}. Trying UDP.")
                         # Try UDP
                         try:
                             req.StreamSetup = {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'UDP'}}
                             # ***** CORRECTION: Removed await/run_in_executor *****
                             stream_info = media_service.GetStreamUri(req)
                             obtained_uri = stream_info.Uri
                             logger.info(f"Got ONVIF stream URI (UDP): {obtained_uri}")
                         except (Fault, TransportError, TimeoutError, ConnectionRefusedError) as e_udp:
                             logger.warning(f"Failed to get UDP stream URI for {self.name}: {e_udp}. Trying profile URI.")
                             rtsp_uri_in_profile = None
                             for attr_name in ['Uri', 'MediaUri', 'RTSPStreamUri', 'StreamUri']:
                                 if hasattr(self.media_profile, attr_name):
                                     uri_val = getattr(self.media_profile, attr_name)
                                     if isinstance(uri_val, str) and uri_val.startswith("rtsp://"):
                                          rtsp_uri_in_profile = uri_val
                                          logger.info(f"Using RTSP URI found directly in profile attribute '{attr_name}': {rtsp_uri_in_profile}")
                                          break
                                     elif isinstance(uri_val, dict) and 'Uri' in uri_val and uri_val['Uri'].startswith("rtsp://"):
                                         rtsp_uri_in_profile = uri_val['Uri']
                                         logger.info(f"Using RTSP URI found in profile dictionary attribute '{attr_name}': {rtsp_uri_in_profile}")
                                         break
                             if rtsp_uri_in_profile:
                                 obtained_uri = rtsp_uri_in_profile
                             else:
                                logger.warning(f"Could not obtain stream URI via GetStreamUri (TCP/UDP) or profile attributes for {self.name}.")

                    # Process obtained URI
                    if obtained_uri:
                        stream_uri = obtained_uri
                        if stream_uri.startswith("rtsp://") and "@" not in stream_uri.split("://")[1].split("/")[0]:
                             if self.user and self.password:
                                 stream_uri = stream_uri.replace("rtsp://", f"rtsp://{self.user}:{self.password}@", 1)
                                 logger.debug(f"Injected credentials into ONVIF URI for {self.name}")
                             elif self.user:
                                 stream_uri = stream_uri.replace("rtsp://", f"rtsp://{self.user}@", 1)
                                 logger.debug(f"Injected username into ONVIF URI for {self.name}")

                    # --- Setup PTZ ---
                    try:
                        self.ptz = self.onvif_cam.create_ptz_service()
                        # ***** CORRECTION: Removed await/run_in_executor *****
                        # ptz_config_options = self.ptz.GetConfigurationOptions(self.media_profile.token) # Removed unused variable
                        # ***** CORRECTION: Removed await/run_in_executor *****
                        ptz_configs = self.ptz.GetConfigurations()

                        found_ptz_token = None
                        if hasattr(self.media_profile, 'PTZConfiguration') and self.media_profile.PTZConfiguration and \
                           hasattr(self.media_profile.PTZConfiguration, 'token'):
                            profile_ptz_token = self.media_profile.PTZConfiguration.token
                            if any(c.token == profile_ptz_token for c in ptz_configs):
                                found_ptz_token = profile_ptz_token
                                logger.info(f"Using PTZ configuration linked in media profile: {found_ptz_token}")

                        if not found_ptz_token and ptz_configs:
                             found_ptz_token = ptz_configs[0].token
                             logger.warning(f"Using first available PTZ configuration: {found_ptz_token} (Media profile link missing or invalid)")

                        if found_ptz_token:
                            self.ptz_configuration_token = found_ptz_token
                        else:
                            logger.warning(f"No PTZ configurations found for camera {self.name}.")
                            self.ptz = None # Disable PTZ

                    except (Fault, TransportError, AttributeError, ConnectionRefusedError, TimeoutError) as e_ptz:
                         logger.warning(f"Could not initialize ONVIF PTZ for {self.name}: {e_ptz}. PTZ disabled.")
                         self.ptz = None
                    except Exception as e_ptz_generic:
                         logger.error(f"Unexpected error during PTZ setup for {self.name}: {e_ptz_generic}", exc_info=True)
                         self.ptz = None

                except (Fault, TransportError, RequestsConnectionError, TimeoutError, ConnectionRefusedError, XMLSyntaxError) as e_onvif:
                    self._set_error(f"ONVIF connection failed: {type(e_onvif).__name__}: {e_onvif}")
                    self.is_connecting = False; return False
                except AttributeError as e_attr:
                     self._set_error(f"ONVIF connection failed (AttributeError): {e_attr}. Check camera compatibility/firmware.")
                     self.is_connecting = False; return False
                except Exception as e_generic:
                    import traceback
                    self._set_error(f"General ONVIF connection error: {e_generic}\n{traceback.format_exc()}")
                    self.is_connecting = False; return False

            # --- End of ONVIF specific block ---

            # --- VideoCapture Initialization ---
            if not stream_uri:
                 if self.is_onvif:
                     self._set_error("Failed to obtain a valid stream URI via ONVIF and no fallback URL configured.")
                 else:
                     self._set_error("No RTSP URL configured for the camera.")
                 self.is_connecting = False
                 return False

            logger.info(f"Opening video capture for {self.name} at URI: {stream_uri}")
            ffmpeg_options = {'rtsp_transport': 'tcp', 'stimeout': '5000000'}
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ";".join(f"{k};{v}" for k, v in ffmpeg_options.items())
            logger.debug(f"Set FFMPEG options: {os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'Not Set')}") # Use get for safety

            try:
                self.cap = cv2.VideoCapture(stream_uri, cv2.CAP_FFMPEG)

                if not self.cap or not self.cap.isOpened():
                    logger.warning(f"VideoCapture failed with TCP hint for {self.name}. Retrying without.")
                    original_options = os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
                    ffmpeg_options_udp = {'rtsp_transport': 'udp', 'stimeout': '5000000'}
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ";".join(f"{k};{v}" for k, v in ffmpeg_options_udp.items())
                    logger.debug(f"Set FFMPEG options (UDP attempt): {os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'Not Set')}")
                    self.cap = cv2.VideoCapture(stream_uri, cv2.CAP_FFMPEG)

                    if not self.cap or not self.cap.isOpened():
                         logger.warning(f"VideoCapture failed with UDP hint for {self.name}. Retrying with default transport.")
                         os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
                         self.cap = cv2.VideoCapture(stream_uri, cv2.CAP_FFMPEG)

                         if not self.cap or not self.cap.isOpened():
                              # Use specific backend identifier if possible
                              backend_name = cv2.videoio_registry.getBackendName(cv2.CAP_FFMPEG)
                              self._set_error(f"Failed to open video stream using cv2.VideoCapture (Backend: {backend_name}) after multiple transport attempts: {stream_uri}")
                              self.is_connecting = False; return False
            finally:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
                logger.debug("Cleared FFMPEG capture options environment variable.")

            # --- Connection Successful ---
            self.is_connected = True
            self.is_connecting = False
            self.last_error = None
            logger.info(f"✅ Successfully connected to camera: {self.name}")
            self.prev_frame_gray = None
            return True

        except Exception as e: # Catch-all for unexpected errors
             import traceback
             self._set_error(f"Unexpected error during connection sequence: {e}\n{traceback.format_exc()}")
             self.is_connecting = False
             return False
    # ***** END OF CORRECTED connect METHOD *****


    def get_frame(self) -> Optional[np.ndarray]:
        """Reads a frame from the video stream. Handles automatic reconnection."""
        if not self.is_connected:
            if not self.connect():
                 return None

        if self.cap is None or not self.cap.isOpened():
             logger.warning(f"get_frame({self.name}): Capture object is invalid/closed unexpectedly. Setting error.")
             self._set_error("VideoCapture object became invalid or closed.")
             return None

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning(f"Frame read failed for {self.name} (ret={ret}, frame is None). Stream might have closed.")
                self._set_error("Frame read failed (stream closed or error).")
                return None

            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        except cv2.error as e:
             logger.error(f"OpenCV error reading frame for {self.name}: {e}")
             self._set_error(f"OpenCV error during frame read: {e}")
             return None
        except Exception as e:
            logger.error(f"Unexpected error reading frame for {self.name}: {e}", exc_info=True)
            self._set_error(f"Unexpected error reading frame: {e}")
            return None

    def detect_motion(self, frame: np.ndarray) -> bool:
        """Detects motion by comparing the current frame to the previous one."""
        if frame is None: return False
        if self.motion_threshold <= 0: return False

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if self.prev_frame_gray is None:
                self.prev_frame_gray = gray
                return False

            frame_delta = cv2.absdiff(self.prev_frame_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            self.prev_frame_gray = gray # Update baseline *after* diff

            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) >= self.motion_threshold:
                     motion_detected = True
                     break # Found significant motion

            return motion_detected

        except cv2.error as e:
            logger.error(f"OpenCV error during motion detection for {self.name}: {e}")
            self.prev_frame_gray = None
            return False
        except Exception as e:
            logger.error(f"Unexpected error during motion detection for {self.name}: {e}", exc_info=True)
            self.prev_frame_gray = None
            return False

    def _perform_ptz_action(self, action_func, action_name: str = "PTZ action", *args) -> bool:
        """Helper to perform a PTZ action with error handling."""
        if not self.is_connected:
             logger.warning(f"Cannot perform {action_name} for {self.name}: Camera not connected.")
             return False
        if not self.ptz:
             logger.warning(f"Cannot perform {action_name} for {self.name}: PTZ not supported or initialized.")
             return False
        if not self.ptz_configuration_token:
             logger.warning(f"Cannot perform {action_name} for {self.name}: PTZ configuration token missing.")
             return False

        try:
            logger.debug(f"Performing {action_name} for {self.name} using token {self.ptz_configuration_token}")
            # Direct synchronous call within the CameraThread
            action_func(self.ptz, self.ptz_configuration_token, *args)
            logger.info(f"{action_name} successful for {self.name}.")
            return True
        except (Fault, TransportError, ConnectionRefusedError, TimeoutError) as e:
            logger.error(f"ONVIF PTZ {action_name} Fault/Error for {self.name}: {e}")
            return False
        except Exception as e:
            logger.error(f"General Error during PTZ {action_name} for {self.name}: {e}", exc_info=True)
            return False

    def move_ptz(self, pan: float, tilt: float, zoom: float):
        """Move the PTZ camera continuously. Values are typically -1.0 to 1.0."""
        def action(ptz_service, token, p, t, z):
            req = ptz_service.create_type('ContinuousMove')
            req.ProfileToken = token
            # Ensure PTZVector and nested Vector types exist (handle potential variations)
            if not hasattr(ptz_service, 'create_type'): # Basic check if service is valid
                logger.error(f"PTZ service for {self.name} seems invalid (missing create_type).")
                return

            try:
                Velocity = ptz_service.create_type('PTZVector')
                Velocity.PanTilt = ptz_service.create_type('Vector2D', x=np.clip(p, -1.0, 1.0), y=np.clip(t, -1.0, 1.0))
                Velocity.Zoom = ptz_service.create_type('Vector1D', x=np.clip(z, -1.0, 1.0))
                req.Velocity = Velocity
                # req.Timeout = 'PT1H' # Optional: Set timeout for the move command
                ptz_service.ContinuousMove(req)
            except Exception as e:
                logger.error(f"Error creating PTZ move request objects for {self.name}: {e}", exc_info=True)
                # Re-raise or handle appropriately? For now, log and let _perform_ptz_action catch higher level error.
                raise

        self._perform_ptz_action(action, "Continuous Move", pan, tilt, zoom)

    def stop_ptz(self):
         """Stops the ongoing PTZ movement."""
         def action(ptz_service, token):
            try:
                req = ptz_service.create_type('Stop')
                req.ProfileToken = token
                req.PanTilt = True # Stop pan/tilt motion
                req.Zoom = True    # Stop zoom motion
                ptz_service.Stop(req)
            except Exception as e:
                logger.error(f"Error creating PTZ stop request object for {self.name}: {e}", exc_info=True)
                raise

         self._perform_ptz_action(action, "Stop")

    def release(self):
        """Releases camera resources."""
        logger.debug(f"Releasing resources for camera: {self.name}")
        self.is_connected = False
        self.release_capture()
        self.onvif_cam = None
        self.ptz = None
        self.media_profile = None

    def __del__(self):
        """Ensure resources are released when the object is destroyed."""
        self.release()

# --- SIEMClient Class ---
class SIEMClient:
    """Handles communication with the Splunk SIEM API."""
    def __init__(self, config: dict):
        self.config = config
        self.session = requests.Session()
        self.api_url = config.get('api_url', '').rstrip('/')
        self.token = config.get('token', '')
        self.query = config.get('query', '')
        self.auth_header_type = config.get("auth_header", "Bearer")
        self.verify_ssl = config.get("verify_ssl", False)

        self.is_configured = bool(self.api_url and self.token and self.query)

        if self.is_configured:
            if self.auth_header_type.lower() == "splunk":
                auth_value = f"Splunk {self.token}"
            else:
                auth_value = f"Bearer {self.token}"
            self.session.headers.update({"Authorization": auth_value})

            if not self.verify_ssl:
                 import urllib3
                 urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.info(f"SIEM Client configured for URL: {self.api_url}, Auth: {self.auth_header_type}")
        else:
            logger.warning("SIEM Client is not fully configured. Alerts will not be fetched.")

    def fetch_alerts(self) -> List[Dict]:
        """Fetches alerts from Splunk using the configured search query."""
        if not self.is_configured: return []

        export_url = f"{self.api_url}/services/search/jobs/export"
        search_query = self.query.strip()
        if not search_query.lower().startswith('search '):
            search_query = f'search {search_query}'

        payload = {"search": search_query, "output_mode": "json"}
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        logger.info(f"Fetching SIEM alerts from {export_url}...")

        try:
            response = self.session.post(
                export_url, data=payload, headers=headers, timeout=45,
                verify=self.verify_ssl, stream=True
            )
            response.raise_for_status()

            alerts = []
            for line in response.iter_lines(decode_unicode=True, delimiter='\n'):
                if line:
                    try:
                        alert_data = json.loads(line)
                        if isinstance(alert_data, dict):
                            if 'result' in alert_data:
                                alerts.append(alert_data['result'])
                            elif '_raw' in alert_data or '_time' in alert_data:
                                alerts.append(alert_data)
                    except json.JSONDecodeError:
                        logger.warning(f"SIEM: Failed to decode JSON line: {line[:150]}...")

            logger.info(f"Successfully fetched {len(alerts)} SIEM alerts.")
            return alerts

        except Timeout:
            logger.error(f"SIEM Error: Request timed out connecting to {export_url}.")
            return []
        except RequestsConnectionError as e:
            logger.error(f"SIEM Error: Connection failed to {export_url}. Check URL/network. Error: {e}")
            return []
        except RequestException as e:
            logger.error(f"SIEM Request Error: {e}")
            if e.response is not None:
                logger.error(f"SIEM Response Status: {e.response.status_code}")
                try: logger.error(f"SIEM Response Body: {e.response.json()}")
                except json.JSONDecodeError: logger.error(f"SIEM Response Body (non-JSON): {e.response.text[:500]}...")
            return []
        except Exception as e:
             import traceback
             logger.error(f"SIEM Error: An unexpected error occurred during fetch: {e}\n{traceback.format_exc()}")
             return []


# ==================== GUI COMPONENTS ====================

class CameraConfigDialog(QDialog):
    """Dialog for adding or editing camera configurations."""
    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Configuration")
        self.setMinimumWidth(450)
        self.config = config or {}

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        form_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self.name_input = QLineEdit(self.config.get('name', ''))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["RTSP URL", "ONVIF"])
        self.url_input = QLineEdit(self.config.get('url', 'rtsp://user:pass@host:port/stream'))
        self.host_input = QLineEdit(self.config.get('host', '192.168.1.100'))
        self.port_input = QLineEdit(str(self.config.get('port', 80)))
        self.user_input = QLineEdit(self.config.get('user', 'admin'))
        self.password_input = QLineEdit(self.config.get('password', ''))
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.motion_thresh_input = QLineEdit(str(self.config.get('motion_threshold', 500)))

        form_layout.addRow("Name*:", self.name_input)
        form_layout.addRow("Type:", self.type_combo)

        self.url_row_widget = QWidget()
        self.url_row_layout = QFormLayout(self.url_row_widget); self.url_row_layout.setContentsMargins(0,0,0,0)
        self.url_label = QLabel("RTSP URL*:")
        self.url_row_layout.addRow(self.url_label, self.url_input)
        form_layout.addRow(self.url_row_widget)

        self.onvif_rows_widget = QWidget()
        self.onvif_rows_layout = QFormLayout(self.onvif_rows_widget); self.onvif_rows_layout.setContentsMargins(0,0,0,0)
        self.host_label = QLabel("ONVIF Host*:")
        self.port_label = QLabel("ONVIF Port*:")
        self.user_label = QLabel("ONVIF User:")
        self.password_label = QLabel("ONVIF Password:")
        self.onvif_rows_layout.addRow(self.host_label, self.host_input)
        self.onvif_rows_layout.addRow(self.port_label, self.port_input)
        self.onvif_rows_layout.addRow(self.user_label, self.user_input)
        self.onvif_rows_layout.addRow(self.password_label, self.password_input)
        form_layout.addRow(self.onvif_rows_widget)

        form_layout.addRow("Motion Threshold:", self.motion_thresh_input)
        self.motion_thresh_input.setToolTip("Contour area threshold (pixels). 0 to disable.")

        layout.addLayout(form_layout)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.type_combo.currentIndexChanged.connect(self.update_fields_visibility)
        is_onvif = self.config.get('onvif', False)
        self.type_combo.setCurrentIndex(1 if is_onvif else 0)
        self.update_fields_visibility()

    def update_fields_visibility(self):
        """Shows/hides fields based on the selected camera type."""
        is_onvif = (self.type_combo.currentText() == "ONVIF")
        self.url_row_widget.setVisible(not is_onvif)
        self.onvif_rows_widget.setVisible(is_onvif)
        self.url_label.setText("RTSP URL*:" if not is_onvif else "RTSP URL:")
        self.host_label.setText("ONVIF Host*:" if is_onvif else "ONVIF Host:")
        self.port_label.setText("ONVIF Port*:" if is_onvif else "ONVIF Port:")

    def get_config(self) -> Optional[Dict]:
        """Validates input and returns the configuration dictionary, or None if invalid."""
        config = {}
        config['name'] = self.name_input.text().strip()
        if not config['name']:
            QMessageBox.warning(self, "Input Error", "Camera Name cannot be empty."); self.name_input.setFocus(); return None

        is_onvif = (self.type_combo.currentText() == "ONVIF")
        config['onvif'] = is_onvif

        if is_onvif:
            config['host'] = self.host_input.text().strip()
            if not config['host']: QMessageBox.warning(self, "Input Error", "ONVIF Host cannot be empty."); self.host_input.setFocus(); return None
            try:
                 port_val = int(self.port_input.text().strip())
                 if not 1 <= port_val <= 65535: raise ValueError("Port out of range")
                 config['port'] = port_val
            except ValueError: QMessageBox.warning(self, "Input Error", "ONVIF Port must be a valid number (1-65535)."); self.port_input.setFocus(); return None
            config['user'] = self.user_input.text().strip()
            config['password'] = self.password_input.text()
            config['url'] = None
        else:
            config['url'] = self.url_input.text().strip()
            if not config['url'] or not config['url'].lower().startswith("rtsp://"):
                 QMessageBox.warning(self, "Input Error", "A valid RTSP URL (starting with rtsp://) is required."); self.url_input.setFocus(); return None
            config['host'] = None; config['port'] = None; config['user'] = None; config['password'] = None

        try:
            mt_val_str = self.motion_thresh_input.text().strip()
            config['motion_threshold'] = int(mt_val_str) if mt_val_str else 0
            if config['motion_threshold'] < 0: raise ValueError("Motion threshold cannot be negative")
        except ValueError: QMessageBox.warning(self, "Input Error", "Motion Threshold must be a non-negative integer."); self.motion_thresh_input.setFocus(); return None

        return config

# --- Camera Processing Thread ---
class CameraThread(QThread):
    """Handles video processing for a camera in a separate thread."""
    new_frame = pyqtSignal(str, object)
    motion_detected_signal = pyqtSignal(str)
    connection_status = pyqtSignal(str, bool, str)

    def __init__(self, camera: SecurityCamera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self._running = True
        self._paused = False
        self._last_emitted_connected_status: Optional[bool] = None
        self._last_emitted_error: Optional[str] = None
        self.setObjectName(f"CameraThread_{self.camera.name}")
        logger.debug(f"Thread {self.objectName()} initialized.")

    def run(self):
        """The main loop executed in the separate thread."""
        logger.info(f"CameraThread started for {self.camera.name}")
        last_status_emit_time = 0
        status_emit_interval = 5 # seconds
        target_fps = 15 # Target FPS for loop timing
        min_sleep = 0.005 # Minimum sleep time

        while self._running:
            if self._paused:
                time.sleep(0.5)
                continue

            loop_start_time = time.time()
            frame = None

            try:
                frame = self.camera.get_frame()

                # Emit Connection Status
                conn_status = self.camera.is_connected
                error_msg = self.camera.last_error
                status_changed = (conn_status != self._last_emitted_connected_status or
                                  error_msg != self._last_emitted_error)
                time_since_last_emit = time.time() - last_status_emit_time

                if status_changed or time_since_last_emit > status_emit_interval:
                    try:
                        self.connection_status.emit(self.camera.name, conn_status, error_msg or "")
                    except RuntimeError as e:
                         logger.warning(f"Error emitting connection status for {self.camera.name} (receiver might be gone): {e}")
                         self._running = False; break
                    self._last_emitted_connected_status = conn_status
                    self._last_emitted_error = error_msg
                    last_status_emit_time = time.time()

                # Process Frame
                if frame is not None:
                    try: self.new_frame.emit(self.camera.name, frame)
                    except RuntimeError as e: logger.warning(f"Error emitting frame for {self.camera.name}: {e}"); self._running = False; break

                    if self.camera.motion_threshold > 0:
                        if self.camera.detect_motion(frame):
                            try: self.motion_detected_signal.emit(self.camera.name)
                            except RuntimeError as e: logger.warning(f"Error emitting motion for {self.camera.name}: {e}"); self._running = False; break

                    # Dynamic sleep to approximate target FPS
                    processing_time = time.time() - loop_start_time
                    sleep_time = max(min_sleep, (1.0 / target_fps) - processing_time)
                    time.sleep(sleep_time)
                else:
                    # Frame retrieval failed
                    time.sleep(1.0) # Wait longer before retrying

            except Exception as e:
                import traceback
                thread_error_msg = f"Unexpected error in CameraThread ({self.camera.name}): {e}\n{traceback.format_exc()}"
                logger.error(thread_error_msg)
                try:
                     if self._last_emitted_error != thread_error_msg:
                         self.connection_status.emit(self.camera.name, False, thread_error_msg)
                         self._last_emitted_connected_status = False
                         self._last_emitted_error = thread_error_msg
                         last_status_emit_time = time.time()
                except RuntimeError: pass
                except Exception as emit_err: logger.error(f"Failed to emit thread loop error status: {emit_err}")
                time.sleep(5.0) # Wait after major error

        logger.info(f"CameraThread stopping for {self.camera.name}...")
        self.camera.release()
        logger.info(f"CameraThread finished for {self.camera.name}")

    def stop(self):
        if self._running: logger.debug(f"Stopping CameraThread for {self.camera.name}..."); self._running = False
        else: logger.debug(f"CameraThread for {self.camera.name} already stopping.")

    def pause(self):
        if not self._paused: logger.debug(f"Pausing CameraThread for {self.camera.name}"); self._paused = True

    def resume(self):
        if self._paused: logger.debug(f"Resuming CameraThread for {self.camera.name}"); self._paused = False

# --- Notification Widget ---
class NotificationManager(QLabel):
    """A custom label for displaying temporary notifications with animation."""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_widget = parent
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QLabel { background-color: rgba(0, 0, 0, 0.8); color: white; border-radius: 6px; padding: 12px 18px; font-size: 10pt; border: 1px solid #555; }")
        self.setWordWrap(True); self.setMinimumWidth(300); self.setMaximumWidth(500)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.MinimumExpanding)
        self.hide()

        self.animation = QPropertyAnimation(self, b"geometry", self)
        self.animation.setDuration(400)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._hide_timer = QTimer(self); self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide_notification)
        self.animation.finished.connect(self._on_animation_finished)

    def show_message(self, message: str, duration: int = 4000, level: str = "info"):
        """Displays a notification message with a specific style and duration."""
        self.setText(message)
        base_style = "color: white; border-radius: 6px; padding: 12px 18px; font-size: 10pt;"
        level_style = ""
        if level == "error": level_style = "background-color: rgba(231, 76, 60, 0.9); border: 1px solid #c0392b;"
        elif level == "warning": level_style = "background-color: rgba(243, 156, 18, 0.9); border: 1px solid #d35400;"
        elif level == "success": level_style = "background-color: rgba(46, 204, 113, 0.9); border: 1px solid #27ae60;"
        else: level_style = "background-color: rgba(52, 152, 219, 0.9); border: 1px solid #2980b9;"
        self.setStyleSheet(f"QLabel {{ {base_style} {level_style} }}")

        self.adjustSize()
        min_height = self.fontMetrics().height() * 2 + 24
        current_height = self.height()
        if current_height < min_height: self.setFixedHeight(min_height)

        parent_width = self.parent_widget.width(); my_width = self.width(); my_height = self.height()
        max_w = parent_width - 40
        if my_width > max_w: my_width = max_w; self.setFixedWidth(my_width)

        start_x = (parent_width - my_width) // 2; start_y = -my_height - 10
        end_x = start_x; end_y = 20
        start_geom = QRect(start_x, start_y, my_width, my_height)
        end_geom = QRect(end_x, end_y, my_width, my_height)

        self.animation.stop(); self._hide_timer.stop()
        self.setGeometry(start_geom); self.show(); self.raise_()
        self.animation.setDirection(QPropertyAnimation.Direction.Forward)
        self.animation.setStartValue(start_geom); self.animation.setEndValue(end_geom)
        self.animation.start()
        self._hide_timer.start(duration)

    def hide_notification(self):
        """Starts the animation to hide the notification."""
        if not self.isVisible() or (self.animation.state() == QPropertyAnimation.State.Running and self.animation.direction() == QPropertyAnimation.Direction.Backward):
            return

        start_geom = self.geometry()
        end_geom = QRect(start_geom.x(), -start_geom.height() - 10, start_geom.width(), start_geom.height())
        self._hide_timer.stop(); self.animation.stop()
        self.animation.setDirection(QPropertyAnimation.Direction.Backward)
        self.animation.setStartValue(start_geom); self.animation.setEndValue(end_geom)
        self.animation.start()

    @pyqtSlot()
    def _on_animation_finished(self):
        """Slot called when animation finishes. Hides widget if it was the hide animation."""
        if self.animation.direction() == QPropertyAnimation.Direction.Backward:
            self.hide()

# --- Camera Marker Item for Map View ---
class CameraMarkerItem(QGraphicsPixmapItem):
    """Represents a draggable camera icon on the map view."""
    markerClicked = pyqtSignal(str)
    markerMoved = pyqtSignal(str, QPointF)

    def __init__(self, camera_name: str, icon: QPixmap, position: QPointF, parent: QGraphicsItem = None):
        super().__init__(icon, parent)
        self.camera_name = camera_name
        self.setPos(position)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.setToolTip(f"Camera: {camera_name}\nClick to view")
        self.setOffset(-icon.width() / 2, -icon.height())
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._is_edit_mode: bool = False
        self._drag_start_pos: QPointF = QPointF()
        shadow = QGraphicsDropShadowEffect(); shadow.setBlurRadius(8)
        shadow.setColor(QColor(0, 0, 0, 100)); shadow.setOffset(2, 2)
        self.setGraphicsEffect(shadow)

    def setEditMode(self, editable: bool):
        if self._is_edit_mode == editable: return
        self._is_edit_mode = editable
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, editable)
        if editable:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.setToolTip(f"Camera: {self.camera_name}\nDrag to move")
        else:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setToolTip(f"Camera: {self.camera_name}\nClick to view")

    def mousePressEvent(self, event: 'QGraphicsSceneMouseEvent'):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._is_edit_mode:
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                self._drag_start_pos = self.pos()
            super().mousePressEvent(event)
        else: event.ignore()

    def mouseMoveEvent(self, event: 'QGraphicsSceneMouseEvent'):
        if self._is_edit_mode and event.buttons() & Qt.MouseButton.LeftButton:
             super().mouseMoveEvent(event)
        else: event.ignore()

    def mouseReleaseEvent(self, event: 'QGraphicsSceneMouseEvent'):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._is_edit_mode:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
                new_pos = self.pos()
                pos_diff = new_pos - self._drag_start_pos
                if pos_diff.manhattanLength() > 0.5:
                    logger.debug(f"Marker '{self.camera_name}' moved from {self._drag_start_pos} to {new_pos}. Emitting signal.")
                    self.markerMoved.emit(self.camera_name, new_pos) # Emit final position
                self.setToolTip(f"Camera: {self.camera_name}\nDrag to move")
            else: # Not edit mode
                 click_threshold = QApplication.startDragDistance()
                 move_dist = (event.screenPos() - event.buttonDownScreenPos(Qt.MouseButton.LeftButton)).manhattanLength()
                 if move_dist < click_threshold:
                      logger.debug(f"Marker '{self.camera_name}' clicked.")
                      self.markerClicked.emit(self.camera_name)
            super().mouseReleaseEvent(event)
        else: event.ignore()

    def hoverEnterEvent(self, event: 'QGraphicsSceneHoverEvent'):
        current_pos = self.pos()
        base_tooltip = self.toolTip().split('\n')[0]
        self.setToolTip(f"{base_tooltip}\nPos: ({current_pos.x():.0f}, {current_pos.y():.0f})")
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: 'QGraphicsSceneHoverEvent'):
        base_tooltip = f"Camera: {self.camera_name}\n"
        base_tooltip += "Drag to move" if self._is_edit_mode else "Click to view"
        self.setToolTip(base_tooltip)
        super().hoverLeaveEvent(event)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: Any) -> Any:
        if self._is_edit_mode and change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            new_pos = value
            scene_rect = self.scene().sceneRect()
            if scene_rect.isValid() and scene_rect.width() > 0 and scene_rect.height() > 0:
                item_rect = self.boundingRect()
                half_width = item_rect.width() / 2
                full_height = item_rect.height()
                constrained_x = max(scene_rect.left() + half_width, min(new_pos.x(), scene_rect.right() - half_width))
                constrained_y = max(scene_rect.top() + full_height, min(new_pos.y(), scene_rect.bottom()))
                constrained_pos = QPointF(constrained_x, constrained_y)
                return constrained_pos
            else: return value
        return super().itemChange(change, value)

# ==================== MAIN APPLICATION WINDOW ====================
class SecurityMonitorApp(QMainWindow):
    """Main application window integrating all components."""
    def __init__(self):
        super().__init__()
        logger.info("Initializing SecurityMonitorApp...")
        self.setWindowTitle("Security Monitor Pro")
        self.setGeometry(100, 100, 1500, 950)
        self.setMinimumSize(1000, 700)
        self._default_app_icon: Optional[QIcon] = self._create_default_icon("app")
        if self._default_app_icon: self.setWindowIcon(self._default_app_icon)

        self.config_filepath: str = "config.yaml"
        self.app_config: Dict[str, Any] = {
            "cameras": [],
            "siem": {"api_url": "", "token": "", "query": "", "auth_header": "Bearer", "verify_ssl": False, "refresh_interval_min": 15},
            "map_view": {"image_path": None, "camera_positions": {}}
        }
        self._settings_dirty: bool = False

        self.cameras: Dict[str, SecurityCamera] = {}
        self.camera_threads: Dict[str, CameraThread] = {}
        self.video_widgets: Dict[str, QLabel] = {}
        self.status_labels: Dict[str, QLabel] = {}
        self.motion_indicators: Dict[str, QLabel] = {}
        self.camera_group_boxes: Dict[str, QGroupBox] = {}
        self.ptz_control_widgets: Dict[str, QWidget] = {}

        self.map_scene: Optional[QGraphicsScene] = None
        self.map_view: Optional[QGraphicsView] = None
        self.map_background_item: Optional[QGraphicsPixmapItem] = None
        self.map_markers: Dict[str, CameraMarkerItem] = {}
        self.map_edit_mode: bool = False
        self._default_camera_icon: Optional[QPixmap] = self._create_default_icon("camera")

        self.siem: Optional[SIEMClient] = None
        self.siem_refresh_timer = QTimer(self)
        self.siem_refresh_timer.timeout.connect(self.refresh_alerts)

        self.load_config(self.config_filepath)
        self.init_ui()
        self.apply_dark_theme()
        self.init_system()
        self.update_siem_timer_interval()
        logger.info("SecurityMonitorApp initialization complete.")

    def update_siem_timer_interval(self):
         """Reads SIEM refresh interval from config and updates the QTimer."""
         siem_refresh_interval_minutes = self.app_config['siem'].get("refresh_interval_min", 15)
         try:
              interval_ms = int(siem_refresh_interval_minutes) * 60 * 1000
              self.siem_refresh_timer.stop()
              if interval_ms > 0:
                  self.siem_refresh_timer.start(interval_ms)
                  logger.info(f"SIEM auto-refresh timer started. Interval: {siem_refresh_interval_minutes} minutes.")
              else: logger.info("SIEM auto-refresh disabled (interval <= 0).")
         except (ValueError, TypeError):
              logger.warning(f"Invalid SIEM refresh interval: '{siem_refresh_interval_minutes}'. Setting to 0 (disabled).")
              self.app_config['siem']['refresh_interval_min'] = 0
              if hasattr(self, 'siem_refresh_input'): self.siem_refresh_input.setText("0")
              self.siem_refresh_timer.stop()

    def _create_default_icon(self, type: str = "app") -> Optional[Union[QIcon, QPixmap]]:
        """Creates a simple default icon or pixmap."""
        try:
            if type == "camera":
                pix = QPixmap(24, 24); pix.fill(Qt.GlobalColor.transparent)
                p = QPainter(pix); p.setRenderHint(QPainter.RenderHint.Antialiasing)
                p.setBrush(QColor(210, 210, 210)); p.setPen(QPen(Qt.GlobalColor.black, 1))
                p.drawRoundedRect(QRectF(2.5, 5.5, 19, 13), 3, 3)
                p.setBrush(QColor(60, 60, 60)); p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(QRectF(7.5, 8.5, 9, 7)); p.end()
                return pix
            else:
                pixmap = QPixmap(32, 32); pixmap.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pixmap); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setPen(QPen(QColor(180, 180, 180), 2)); painter.setBrush(QColor(70, 70, 70))
                painter.drawRoundedRect(QRectF(3.5, 3.5, 25, 25), 5, 5)
                painter.setPen(QPen(QColor(60, 180, 230), 3))
                painter.drawLine(QPointF(8, 16), QPointF(24, 16)); painter.drawPoint(QPointF(16, 11))
                painter.end()
                return QIcon(pixmap)
        except Exception as e: logger.error(f"Error creating default '{type}' icon: {e}"); return None

    def load_config(self, filepath: str):
        """Loads application configuration from a YAML file."""
        logger.info(f"Attempting to load configuration from: {filepath}")
        self.config_filepath = filepath
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                    if loaded_config and isinstance(loaded_config, dict):
                        def merge_dicts(default, loaded):
                             merged = default.copy()
                             for key, value in loaded.items():
                                 if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                                     merged[key] = merge_dicts(merged[key], value)
                                 elif isinstance(merged.get(key), list) and isinstance(value, list):
                                      merged[key] = value # Overwrite lists
                                 else: merged[key] = value
                             return merged
                        self.app_config = merge_dicts(self.app_config, loaded_config)
                        logger.info(f"Configuration successfully loaded from {filepath}")
                        self._validate_and_correct_config()
                    elif loaded_config is None: logger.warning(f"Config file {filepath} is empty. Using defaults.")
                    else: logger.error(f"Config file {filepath} invalid. Using defaults.")
            else: logger.info(f"Config file {filepath} not found. Using defaults.")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file {filepath}: {e}", exc_info=True)
            if hasattr(self, 'central_widget') and self.central_widget.isVisible():
                 QMessageBox.critical(self, "Config Error", f"Error parsing config file:\n{filepath}\n\n{e}\n\nUsing defaults.")
        except Exception as e:
            logger.error(f"Error loading config from {filepath}: {e}", exc_info=True)
            if hasattr(self, 'central_widget') and self.central_widget.isVisible():
                 QMessageBox.critical(self, "Config Error", f"Error loading config file:\n{filepath}\n\n{e}\n\nUsing defaults.")

    def _validate_and_correct_config(self):
        """Performs basic validation and type correction on the loaded app_config."""
        if not isinstance(self.app_config.get('cameras'), list): self.app_config['cameras'] = []
        siem_conf = self.app_config.get('siem', {}); self.app_config['siem'] = siem_conf
        siem_conf['api_url'] = str(siem_conf.get('api_url', ''))
        siem_conf['token'] = str(siem_conf.get('token', ''))
        siem_conf['query'] = str(siem_conf.get('query', ''))
        siem_conf['auth_header'] = str(siem_conf.get('auth_header', 'Bearer'))
        siem_conf['verify_ssl'] = bool(siem_conf.get('verify_ssl', False))
        try: siem_conf['refresh_interval_min'] = int(siem_conf.get('refresh_interval_min', 15))
        except: siem_conf['refresh_interval_min'] = 15
        map_conf = self.app_config.get('map_view', {}); self.app_config['map_view'] = map_conf
        map_conf['image_path'] = str(map_conf.get('image_path')) if map_conf.get('image_path') else None
        if not isinstance(map_conf.get('camera_positions'), dict): map_conf['camera_positions'] = {}
        for name, pos in list(map_conf['camera_positions'].items()):
            if not isinstance(pos, dict) or 'x' not in pos or 'y' not in pos: del map_conf['camera_positions'][name]; continue
            try: pos['x'] = float(pos['x']); pos['y'] = float(pos['y'])
            except: del map_conf['camera_positions'][name]
        valid_cameras = []
        for i, cam_conf in enumerate(self.app_config['cameras']):
            if not isinstance(cam_conf, dict): continue
            if 'name' not in cam_conf or not str(cam_conf['name']).strip(): cam_conf['name'] = f"Camera_{i+1}_{int(time.time())}"
            cam_conf['name'] = str(cam_conf['name']).strip()
            cam_conf['onvif'] = bool(cam_conf.get('onvif', False))
            cam_conf['url'] = str(cam_conf.get('url')) if cam_conf.get('url') else None
            cam_conf['host'] = str(cam_conf.get('host')) if cam_conf.get('host') else None
            try: cam_conf['port'] = int(cam_conf.get('port')) if cam_conf.get('port') is not None else None
            except: cam_conf['port'] = 80 if cam_conf['onvif'] else None
            cam_conf['user'] = str(cam_conf.get('user')) if cam_conf.get('user') else None
            cam_conf['password'] = str(cam_conf.get('password')) if cam_conf.get('password') else None
            try: cam_conf['motion_threshold'] = int(cam_conf.get('motion_threshold', 500))
            except: cam_conf['motion_threshold'] = 500
            if cam_conf['motion_threshold'] < 0: cam_conf['motion_threshold'] = 0
            valid_cameras.append(cam_conf)
        self.app_config['cameras'] = valid_cameras

    def save_config(self, filepath: Optional[str] = None) -> bool:
        """Saves the current application configuration to a YAML file."""
        save_path = filepath or self.config_filepath
        logger.info(f"Attempting to save configuration to: {save_path}")
        try:
            if self.map_markers:
                 current_marker_positions = {}
                 active_camera_names = {cam.get('name') for cam in self.app_config.get('cameras', [])}
                 for name, marker in self.map_markers.items():
                     if name in active_camera_names:
                          pos = marker.pos()
                          current_marker_positions[name] = {'x': round(pos.x(), 2), 'y': round(pos.y(), 2)}
                     else: logger.warning(f"Map marker found for non-existent camera '{name}'. Position not saved.")
                 if 'map_view' not in self.app_config: self.app_config['map_view'] = {}
                 self.app_config['map_view']['camera_positions'] = current_marker_positions

            save_dir = os.path.dirname(save_path)
            if save_dir: os.makedirs(save_dir, exist_ok=True) # Ensure directory exists

            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.app_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True, indent=2)
            logger.info(f"Configuration successfully saved to {save_path}")
            self._settings_dirty = False
            self.update_window_title()
            return True
        except yaml.YAMLError as e:
            logger.error(f"YAML error saving configuration to {save_path}: {e}", exc_info=True)
            if hasattr(self, 'notifications'): self.notifications.show_message(f"Error saving config (YAML format error): {e}", level="error")
            return False
        except Exception as e:
            logger.error(f"Error saving configuration to {save_path}: {e}", exc_info=True)
            if hasattr(self, 'notifications'): self.notifications.show_message(f"Error saving config: {e}", level="error")
            return False

    def mark_settings_dirty(self, dirty=True):
         """Marks the settings as changed and updates window title."""
         if self._settings_dirty != dirty: self._settings_dirty = dirty; self.update_window_title()

    def update_window_title(self):
         """Updates the main window title, adding '*' if settings are dirty."""
         base_title = "Security Monitor Pro"
         config_filename = os.path.basename(self.config_filepath) if self.config_filepath else "Untitled"
         title = f"{base_title} - {config_filename}"
         if self._settings_dirty: title += " *"
         self.setWindowTitle(title)

    def init_ui(self):
        """Initialize the main user interface elements."""
        logger.debug("Initializing UI...")
        self.update_window_title()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget); self.main_layout.setContentsMargins(5, 5, 5, 5); self.main_layout.setSpacing(5)
        self.init_menu_bar()
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Initializing UI...")
        self.notifications = NotificationManager(self.central_widget)
        self.tabs = QTabWidget(); self.tabs.currentChanged.connect(self.on_tab_changed); self.main_layout.addWidget(self.tabs)
        self.create_monitor_tab(); self.create_map_view_tab(); self.create_settings_tab()
        self.tabs.setCurrentIndex(0); self.status_bar.showMessage("UI Initialized.", 3000)
        logger.debug("UI Initialization complete.")

    def init_menu_bar(self):
        """Sets up the main menu bar."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        load_action = QAction(QIcon.fromTheme("document-open", self._create_default_icon()), "Load Config...", self); load_action.setShortcut("Ctrl+O"); load_action.triggered.connect(self.load_config_dialog); file_menu.addAction(load_action)
        save_action = QAction(QIcon.fromTheme("document-save", self._create_default_icon()), "Save Config", self); save_action.setShortcut("Ctrl+S"); save_action.triggered.connect(lambda: self.save_config()); file_menu.addAction(save_action)
        save_as_action = QAction(QIcon.fromTheme("document-save-as", self._create_default_icon()), "Save Config As...", self); save_as_action.triggered.connect(self.save_config_dialog); file_menu.addAction(save_as_action)
        file_menu.addSeparator()
        exit_action = QAction(QIcon.fromTheme("application-exit", self._create_default_icon()), "Exit", self); exit_action.setShortcut("Ctrl+Q"); exit_action.triggered.connect(self.close); file_menu.addAction(exit_action)
        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction(QIcon.fromTheme("help-about", self._create_default_icon()), "About", self); about_action.triggered.connect(self.show_about); help_menu.addAction(about_action)

    def on_tab_changed(self, index: int):
        """Slot called when the current tab changes."""
        logger.debug(f"Switched to tab index {index}: {self.tabs.tabText(index)}")

    def load_config_dialog(self):
         """Opens file dialog to load a configuration file."""
         if self.check_unsaved_changes("load a new configuration file"):
             current_dir = os.path.dirname(self.config_filepath) if self.config_filepath else ""
             filepath, _ = QFileDialog.getOpenFileName(self, "Load Configuration", current_dir, "YAML Files (*.yaml *.yml);;All Files (*)")
             if filepath:
                 self.load_config(filepath)
                 self.refresh_settings_ui()
                 self.stop_all_cameras()
                 self.init_system()
                 self.recreate_monitor_tab()
                 self.load_map_image()
                 self.update_siem_timer_interval()
                 self.mark_settings_dirty(False)
                 self.notifications.show_message(f"Configuration loaded from {os.path.basename(filepath)}.", level="success")

    def save_config_dialog(self):
         """Opens file dialog to save configuration to a new file."""
         start_path = self.config_filepath or "config.yaml"
         filepath, _ = QFileDialog.getSaveFileName(self, "Save Configuration As", start_path, "YAML Files (*.yaml *.yml);;All Files (*)")
         if filepath:
            if self.save_config(filepath):
                 self.config_filepath = filepath
                 self.mark_settings_dirty(False)
                 self.notifications.show_message(f"Configuration saved to {os.path.basename(filepath)}", level="success")

    def stop_all_cameras(self):
         """Stops all running camera threads gracefully."""
         logger.info(f"Stopping all ({len(self.camera_threads)}) camera threads...")
         threads_to_stop = list(self.camera_threads.values())
         if not threads_to_stop: logger.info("No camera threads to stop.")
         else:
             for thread in threads_to_stop:
                 if thread and thread.isRunning(): thread.stop()
             start_wait = time.time(); max_wait_sec = 10.0
             threads_still_running = threads_to_stop[:]
             while threads_still_running and (time.time() - start_wait) < max_wait_sec:
                  threads_still_running = [t for t in threads_still_running if t and t.isRunning()]
                  if threads_still_running: QApplication.processEvents(); time.sleep(0.1)
                  else: break
             if threads_still_running: logger.warning(f"{len(threads_still_running)} threads did not stop gracefully within {max_wait_sec}s.")
             else: logger.info("All camera threads stopped gracefully.")

         self.cameras.clear(); self.camera_threads.clear()
         self.video_widgets.clear(); self.status_labels.clear(); self.motion_indicators.clear()
         self.camera_group_boxes.clear(); self.ptz_control_widgets.clear()
         if self.map_scene:
             for marker in list(self.map_markers.values()): self.map_scene.removeItem(marker)
         self.map_markers.clear()
         logger.info("Cleared camera state dictionaries and map markers.")

    def init_system(self):
        """Initializes cameras and SIEM client based on current app_config."""
        logger.info("Initializing system components (Cameras, SIEM)...")
        if hasattr(self, 'status_bar'): self.status_bar.showMessage("Initializing system components...")
        QApplication.processEvents()
        self.stop_all_cameras() # Stop and clear existing first

        camera_init_errors = []; unique_names = set()
        for i, config in enumerate(self.app_config.get('cameras', [])):
            cam_name = config.get('name', f'UnnamedCamera_{i}')
            if cam_name in unique_names: logger.error(f"Duplicate camera name '{cam_name}'! Skipping."); camera_init_errors.append(f"Duplicate: {cam_name}"); continue
            unique_names.add(cam_name)
            try:
                logger.debug(f"Creating SecurityCamera for '{cam_name}'")
                camera = SecurityCamera(config); self.cameras[cam_name] = camera
                logger.debug(f"Creating CameraThread for '{cam_name}'")
                thread = CameraThread(camera, self)
                thread.new_frame.connect(self.update_video_frame, Qt.ConnectionType.QueuedConnection)
                thread.motion_detected_signal.connect(self.on_motion_detected, Qt.ConnectionType.QueuedConnection)
                thread.connection_status.connect(self.on_camera_connection_status, Qt.ConnectionType.QueuedConnection)
                self.camera_threads[cam_name] = thread; thread.start()
            except Exception as e:
                import traceback; err_msg = f"Failed to initialize camera '{cam_name}': {e}\n{traceback.format_exc()}"
                logger.error(err_msg); camera_init_errors.append(f"{cam_name}: Init Error")
                if cam_name in self.cameras: del self.cameras[cam_name]
                if cam_name in self.camera_threads: thread = self.camera_threads.pop(cam_name); thread.stop(); thread.wait(500)

        try:
            logger.debug("Initializing SIEMClient...")
            self.siem = SIEMClient(self.app_config['siem'])
            if self.siem.is_configured: logger.info("SIEM client initialized."); QTimer.singleShot(1500, self.refresh_alerts)
            else: logger.warning("SIEM client not configured."); self.alerts_display.setPlainText("SIEM client not configured.") if hasattr(self, 'alerts_display') else None
        except Exception as e: err_msg = f"Failed to initialize SIEM client: {e}"; logger.error(err_msg, exc_info=True); self.notifications.show_message(f"SIEM Init Error: {e}", level="error") if hasattr(self, 'notifications') else None

        self.update_map_markers() # Update markers based on (potentially empty) camera list

        status_msg = "System ready."; num_cameras = len(self.cameras)
        if camera_init_errors: status_msg = f"{num_cameras} camera(s) active. System initialized with {len(camera_init_errors)} error(s)."; self.notifications.show_message(f"Init Issues: {'; '.join(camera_init_errors)}", level="warning") if hasattr(self, 'notifications') else None
        elif num_cameras == 0: status_msg = "System ready. No cameras configured."
        else: status_msg = f"System ready. {num_cameras} camera(s) active."
        if hasattr(self, 'status_bar'): self.status_bar.showMessage(status_msg, 5000)
        logger.info(f"System initialization finished. Status: {status_msg}")

    @pyqtSlot()
    def refresh_alerts(self):
        """Fetches SIEM alerts and updates the display using a worker thread."""
        if not hasattr(self, 'alerts_display') or self.alerts_display is None: return
        if not self.siem or not self.siem.is_configured: self.alerts_display.setPlainText("SIEM client not configured."); return

        logger.debug("Refreshing SIEM alerts (background thread)...")
        self.alerts_display.setPlainText("Fetching SIEM alerts..."); QApplication.processEvents()

        def fetch_task():
            alerts = self.siem.fetch_alerts()
            QMetaObject.invokeMethod(self, "_update_alerts_display", Qt.ConnectionType.QueuedConnection, Q_ARG(list, alerts))

        import threading
        fetch_thread = threading.Thread(target=fetch_task, daemon=True); fetch_thread.start()

    @pyqtSlot(list)
    def _update_alerts_display(self, alerts: List[Dict]):
        """Updates the QTextEdit with formatted SIEM alerts. Called via invokeMethod."""
        if not hasattr(self, 'alerts_display') or self.alerts_display is None: return
        logger.debug(f"Updating alerts display with {len(alerts)} alerts.")
        self.alerts_display.clear()
        if not alerts: self.alerts_display.setPlainText("No alerts found matching the criteria."); return

        html_parts = [
            "<style>p{margin-bottom:3px;line-height:1.3;}b{color:#aaddff;}hr{border:none;border-top:1px solid #555;margin:5px 0;}"
            ".raw{font-family:Consolas,'Courier New',monospace;white-space:pre-wrap;color:#ccc;font-size:8pt;display:block;background-color:#2f2f2f;padding:3px;border-radius:3px;}</style>"
        ]
        max_alerts_display = 150; alerts_to_display = alerts[:max_alerts_display]
        for alert in alerts_to_display:
            html_parts.append("<p>")
            timestamp = alert.get('_time', ''); ts_display = timestamp
            if timestamp:
                try:
                    if isinstance(timestamp, (int, float)): dt_obj = datetime.datetime.fromtimestamp(timestamp); ts_display = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                         ts_str = str(timestamp).strip().split('+')[0].replace('Z', '').split('.')[0]
                         for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"]:
                             try: dt_obj = datetime.datetime.strptime(ts_str, fmt); ts_display = dt_obj.strftime('%Y-%m-%d %H:%M:%S'); break
                             except ValueError: continue
                except Exception: pass
                html_parts.append(f"<b>Time:</b> {ts_display}<br>")
            host = alert.get('host', ''); source = alert.get('source', ''); sourcetype = alert.get('sourcetype', '')
            if host: html_parts.append(f"<b>Host:</b> {host}<br>")
            if sourcetype: html_parts.append(f"<b>Type:</b> {sourcetype}<br>")
            elif source: html_parts.append(f"<b>Source:</b> {source}<br>")
            raw_event = str(alert.get('_raw', 'No raw event.'))
            escaped_event = raw_event.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            max_raw_len = 600; display_event = escaped_event[:max_raw_len] + ('...' if len(escaped_event) > max_raw_len else '')
            html_parts.append(f"<span class='raw'>{display_event}</span></p><hr>")
        if len(alerts) > max_alerts_display: html_parts.append(f"<p><i>(Showing first {max_alerts_display} of {len(alerts)} alerts)</i></p>")
        self.alerts_display.setHtml("".join(html_parts)); self.alerts_display.moveCursor(QTextCursor.MoveOperation.Start)
        if hasattr(self, 'status_bar'): self.status_bar.showMessage(f"SIEM alerts refreshed: {len(alerts)} found.", 4000)

    def recreate_monitor_tab(self):
        """Removes and recreates the monitor tab UI elements."""
        logger.info("Recreating monitor tab...")
        monitor_tab_index = -1
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Monitoring": monitor_tab_index = i; break
        if monitor_tab_index != -1:
            widget_to_remove = self.tabs.widget(monitor_tab_index)
            if widget_to_remove:
                 current_tab_index = self.tabs.currentIndex()
                 self.tabs.removeTab(monitor_tab_index); widget_to_remove.deleteLater()
                 logger.debug("Removed existing monitor tab widget.")
            else: logger.warning("Could not find widget for monitor tab to remove.")
            self.video_widgets.clear(); self.status_labels.clear(); self.motion_indicators.clear()
            self.camera_group_boxes.clear(); self.ptz_control_widgets.clear()
            logger.debug("Cleared monitor tab widget caches.")
            self.create_monitor_tab(); logger.debug("Created new monitor tab content.")
            new_monitor_tab_index = 0
            if current_tab_index == monitor_tab_index: self.tabs.setCurrentIndex(new_monitor_tab_index)
            elif 0 <= current_tab_index < monitor_tab_index : self.tabs.setCurrentIndex(current_tab_index)
            elif current_tab_index > monitor_tab_index: self.tabs.setCurrentIndex(current_tab_index - 1)
            else: self.tabs.setCurrentIndex(new_monitor_tab_index)
        else: logger.warning("Monitor tab not found, creating anew."); self.create_monitor_tab(); self.tabs.setCurrentIndex(0)

    def create_monitor_tab(self):
        """Creates the monitoring tab with camera feeds and alerts."""
        monitor_tab = QWidget(); main_hbox = QHBoxLayout(monitor_tab)
        main_hbox.setContentsMargins(5, 5, 5, 5); main_hbox.setSpacing(10)
        camera_scroll_area = QScrollArea(); camera_scroll_area.setWidgetResizable(True)
        camera_scroll_area.setStyleSheet("QScrollArea { border: none; }")
        camera_area_container = QWidget(); camera_layout = QVBoxLayout(camera_area_container)
        camera_layout.setContentsMargins(0, 0, 0, 0); camera_layout.setSpacing(10)
        self.camera_group_boxes.clear(); self.video_widgets.clear(); self.status_labels.clear()
        self.motion_indicators.clear(); self.ptz_control_widgets.clear()
        configured_cameras = self.app_config.get('cameras', [])
        if not configured_cameras:
             no_cam_label = QLabel("No cameras configured.\nAdd cameras via Settings tab."); no_cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter); no_cam_label.setStyleSheet("color: #aaa;")
             camera_layout.addWidget(no_cam_label, 0, Qt.AlignmentFlag.AlignCenter)
        else:
             logger.debug(f"Creating widgets for {len(configured_cameras)} cameras...")
             for config in configured_cameras:
                if camera_box := self._create_camera_widget(config): self.camera_group_boxes[config['name']] = camera_box; camera_layout.addWidget(camera_box)
                else: logger.error(f"Failed to create widget for camera: {config.get('name')}")
             camera_layout.addStretch()
        camera_scroll_area.setWidget(camera_area_container)

        alerts_panel = QGroupBox("Security Information & Event Management (SIEM)")
        alerts_layout = QVBoxLayout(alerts_panel); alerts_layout.setContentsMargins(8, 8, 8, 8)
        if not hasattr(self, 'alerts_display') or self.alerts_display is None:
             logger.debug("Creating SIEM alerts_display widget.")
             self.alerts_display = QTextEdit(); self.alerts_display.setReadOnly(True)
             self.alerts_display.setStyleSheet("background-color:#262626;color:#ddd;border:1px solid #555;font-family:Consolas,'Courier New',monospace;font-size:9pt;")
             self.alerts_display.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.refresh_btn = QPushButton(QIcon.fromTheme("view-refresh", self._create_default_icon()), " Refresh Alerts")
        self.refresh_btn.setToolTip("Manually fetch latest SIEM alerts"); self.refresh_btn.clicked.connect(self.refresh_alerts)
        self.refresh_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        alerts_layout.addWidget(self.alerts_display, 1); alerts_layout.addWidget(self.refresh_btn, 0, Qt.AlignmentFlag.AlignRight)

        main_hbox.addWidget(camera_scroll_area, 65); main_hbox.addWidget(alerts_panel, 35)
        self.tabs.insertTab(0, monitor_tab, QIcon.fromTheme("video-display", self._create_default_icon("app")), "Monitoring")
        logger.debug("Monitor tab created and inserted at index 0.")

    def _create_camera_widget(self, config: dict) -> Optional[QGroupBox]:
        """Creates a QGroupBox containing widgets for a single camera."""
        camera_name = config.get('name')
        if not camera_name: return None
        camera_box = QGroupBox(camera_name); camera_box_layout = QVBoxLayout(camera_box)
        camera_box_layout.setContentsMargins(5, 8, 5, 5); camera_box_layout.setSpacing(4)
        top_hbox = QHBoxLayout(); top_hbox.setSpacing(8)
        video_label = QLabel("Initializing..."); video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_label.setStyleSheet("background-color:#1e1e1e;color:#888;border:1px solid #444;border-radius:3px;")
        video_label.setMinimumSize(320, 180); video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_widgets[camera_name] = video_label; top_hbox.addWidget(video_label, 1)
        status_vbox = QVBoxLayout(); status_vbox.setSpacing(5); status_vbox.setAlignment(Qt.AlignmentFlag.AlignTop)
        status_label = QLabel("⚪ Waiting..."); status_label.setToolTip("Camera connection status"); status_label.setStyleSheet("font-size:8pt;color:#aaa;")
        self.status_labels[camera_name] = status_label; status_vbox.addWidget(status_label)
        motion_label = QLabel(); motion_label.setToolTip("Motion Detection Status"); motion_label.setFixedSize(16, 16)
        motion_label.setStyleSheet("background-color:transparent;border:1px solid #666;border-radius:8px;")
        self.motion_indicators[camera_name] = motion_label; status_vbox.addWidget(motion_label, 0, Qt.AlignmentFlag.AlignLeft)
        top_hbox.addLayout(status_vbox); camera_box_layout.addLayout(top_hbox, 1)
        controls_hbox = QHBoxLayout(); controls_hbox.setSpacing(5); controls_hbox.setContentsMargins(0, 5, 0, 0)
        snapshot_btn = QPushButton(QIcon.fromTheme("camera-photo", self._create_default_icon("app")), "")
        snapshot_btn.setToolTip(f"Take Snapshot ({camera_name})"); snapshot_btn.setFixedSize(30, 30); snapshot_btn.setIconSize(QSize(18, 18))
        snapshot_btn.clicked.connect(lambda checked=False, name=camera_name: self.take_snapshot(name)); controls_hbox.addWidget(snapshot_btn)
        controls_hbox.addSpacing(15)
        ptz_widget = QWidget(); ptz_layout = QHBoxLayout(ptz_widget); ptz_layout.setContentsMargins(0,0,0,0); ptz_layout.setSpacing(2)
        ptz_widget._ptz_buttons: List[QPushButton] = []
        if config.get('onvif', False):
            ptz_buttons_map = self._create_ptz_controls(camera_name); ptz_widget._ptz_buttons = list(ptz_buttons_map.values())
            ptz_layout.addWidget(ptz_buttons_map["left"]); ptz_layout.addWidget(ptz_buttons_map["up"])
            ptz_layout.addWidget(ptz_buttons_map["down"]); ptz_layout.addWidget(ptz_buttons_map["right"])
            ptz_layout.addSpacing(10); ptz_layout.addWidget(ptz_buttons_map["zoomin"]); ptz_layout.addWidget(ptz_buttons_map["zoomout"])
            ptz_widget.setVisible(True)
        else: ptz_widget.setVisible(False)
        self.ptz_control_widgets[camera_name] = ptz_widget; controls_hbox.addWidget(ptz_widget)
        controls_hbox.addStretch(); camera_box_layout.addLayout(controls_hbox)
        return camera_box

    def _create_ptz_controls(self, camera_name: str) -> Dict[str, QPushButton]:
        """Creates a dictionary of PTZ control buttons for a camera."""
        buttons: Dict[str, QPushButton] = {}; ptz_button_size = QSize(28, 28); ptz_icon_size = QSize(16, 16)
        def create_ptz_button(key: str, icon_name: str, tooltip: str, pressed_action, released_action) -> QPushButton:
             icon = QIcon.fromTheme(icon_name, QIcon()); button = QPushButton(icon, "")
             button.setToolTip(f"{tooltip} ({camera_name})"); button.setFixedSize(ptz_button_size); button.setIconSize(ptz_icon_size)
             button.setAutoRepeat(False); button.pressed.connect(pressed_action); button.released.connect(released_action); button.setEnabled(False)
             buttons[key] = button; return button
        ptz_speed = 0.6; zoom_speed = 0.5
        create_ptz_button("up", "go-up", "Tilt Up", lambda: self.start_ptz(camera_name, 0, ptz_speed, 0), lambda: self.stop_ptz(camera_name))
        create_ptz_button("down", "go-down", "Tilt Down", lambda: self.start_ptz(camera_name, 0, -ptz_speed, 0), lambda: self.stop_ptz(camera_name))
        create_ptz_button("left", "go-previous", "Pan Left", lambda: self.start_ptz(camera_name, -ptz_speed, 0, 0), lambda: self.stop_ptz(camera_name))
        create_ptz_button("right", "go-next", "Pan Right", lambda: self.start_ptz(camera_name, ptz_speed, 0, 0), lambda: self.stop_ptz(camera_name))
        create_ptz_button("zoomin", "zoom-in", "Zoom In", lambda: self.start_ptz(camera_name, 0, 0, zoom_speed), lambda: self.stop_ptz(camera_name))
        create_ptz_button("zoomout", "zoom-out", "Zoom Out", lambda: self.start_ptz(camera_name, 0, 0, -zoom_speed), lambda: self.stop_ptz(camera_name))
        return buttons

    @pyqtSlot(str, object)
    def update_video_frame(self, camera_name: str, frame: Any):
        """Updates the video label with a new frame."""
        if camera_name not in self.video_widgets: return
        video_label = self.video_widgets[camera_name]
        if not isinstance(frame, np.ndarray) or frame.size == 0: return
        try:
            h, w, ch = frame.shape
            if ch == 3:
                q_img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                if pixmap.isNull(): logger.warning(f"Failed QPixmap creation for {camera_name}"); return
                label_size = video_label.size()
                if label_size.isValid() and label_size.width() > 10 and label_size.height() > 10:
                    scaled_pixmap = pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    video_label.setPixmap(scaled_pixmap)
                    if video_label.text(): video_label.setText("")
                    video_label.setStyleSheet("background-color:#111;border:1px solid #444;border-radius:3px;")
            else: logger.warning(f"Frame with unexpected channels ({ch}) for {camera_name}")
        except Exception as e:
             logger.error(f"Error updating video frame for {camera_name}: {e}", exc_info=True)
             video_label.setText(f"Frame Error\n{type(e).__name__}"); video_label.setStyleSheet("background-color:#300;color:red;border:1px solid red;")

    @pyqtSlot(str, bool, str)
    def on_camera_connection_status(self, camera_name: str, is_connected: bool, error_message: str):
         """Updates UI elements based on camera connection status changes."""
         logger.debug(f"Status for '{camera_name}': Connected={is_connected}, Err='{error_message[:50]}...'")
         if camera_name not in self.status_labels or camera_name not in self.video_widgets or \
            camera_name not in self.camera_group_boxes or camera_name not in self.ptz_control_widgets:
             logger.warning(f"Status update for '{camera_name}' skipped: UI widgets not found."); return

         status_label = self.status_labels[camera_name]; video_label = self.video_widgets[camera_name]
         group_box = self.camera_group_boxes[camera_name]; ptz_widget = self.ptz_control_widgets[camera_name]
         base_title = camera_name

         if is_connected:
             status_label.setText("🟢 Connected"); status_label.setStyleSheet("font-size:8pt;color:#4CAF50;font-weight:bold;")
             status_label.setToolTip("Camera connected."); group_box.setTitle(base_title)
             if video_label.text() and not video_label.pixmap():
                  video_label.setText(""); video_label.setStyleSheet("background-color:#1e1e1e;color:#888;border:1px solid #444;border-radius:3px;")
         else:
             status_label.setText("🔴 Disconnected"); status_label.setStyleSheet("font-size:8pt;color:#F44336;font-weight:bold;")
             tooltip = f"Disconnected.\n{error_message or 'No specific error reported.'}".strip()
             status_label.setToolTip(tooltip); group_box.setTitle(f"{base_title} (Offline)")
             current_pixmap = video_label.pixmap()
             if current_pixmap is None or current_pixmap.isNull():
                 display_error = error_message if error_message else "Connection failed."
                 if len(display_error) > 100: display_error = display_error[:97] + "..."
                 video_label.setText(f"Disconnected\n({display_error})"); video_label.setStyleSheet("background-color:#333;color:#aaa;border:1px solid #555;border-radius:3px;")
                 video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

         can_ptz = False; camera_instance = self.cameras.get(camera_name)
         if camera_instance and camera_instance.is_onvif:
             ptz_widget.setVisible(True)
             can_ptz = is_connected and camera_instance.ptz is not None
             ptz_tooltip_suffix = ""
             if not is_connected: ptz_tooltip_suffix = " (Camera Offline)"
             elif camera_instance.ptz is None: ptz_tooltip_suffix = " (PTZ Not Available)"
             ptz_buttons = getattr(ptz_widget, '_ptz_buttons', [])
             for button in ptz_buttons:
                 button.setEnabled(can_ptz)
                 base_tooltip = button.toolTip().split(' (')[0]; button.setToolTip(base_tooltip + ptz_tooltip_suffix)
         else: ptz_widget.setVisible(False)

    @pyqtSlot(str)
    def on_motion_detected(self, camera_name: str):
        """Highlights the motion indicator when motion is detected."""
        if camera_name in self.motion_indicators:
             indicator = self.motion_indicators[camera_name]
             indicator.setStyleSheet("background-color:#ffdd00;border:1px solid #ffaa00;border-radius:8px;")
             QTimer.singleShot(1200, lambda name=camera_name: self._reset_motion_indicator(name))
        if camera_name in self.camera_group_boxes:
             self.highlight_widget(self.camera_group_boxes[camera_name], duration_ms=1500, color=QColor("#ffdd00"))

    def _reset_motion_indicator(self, camera_name: str):
         """Resets the motion indicator back to its default state."""
         if camera_name in self.motion_indicators:
             indicator = self.motion_indicators.get(camera_name)
             if indicator: indicator.setStyleSheet("background-color:transparent;border:1px solid #666;border-radius:8px;")

    @pyqtSlot(str)
    def take_snapshot(self, camera_name: str):
        """Saves the current frame from the specified camera to a file."""
        logger.info(f"Snapshot requested for camera: {camera_name}")
        if camera_name not in self.video_widgets: logger.warning(f"No video widget for '{camera_name}'."); self.notifications.show_message(f"Cannot snapshot: UI not found.", level="error"); return
        video_label = self.video_widgets[camera_name]; pixmap = video_label.pixmap()
        if not pixmap or pixmap.isNull(): logger.warning(f"No valid image for {camera_name}."); self.notifications.show_message(f"Cannot snapshot: No image available.", level="warning"); return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_cam_name = "".join(c if c.isalnum() else "_" for c in camera_name).strip('_')
        default_filename = f"{safe_cam_name}_snapshot_{timestamp}.png"
        snapshot_dir = "snapshots"
        try: os.makedirs(snapshot_dir, exist_ok=True)
        except OSError as e: logger.error(f"Error creating dir '{snapshot_dir}': {e}"); self.notifications.show_message(f"Snapshot Error: Cannot create directory.", level="error"); return
        default_path = os.path.join(snapshot_dir, default_filename)

        file_path, selected_filter = QFileDialog.getSaveFileName(self, f"Save Snapshot - {camera_name}", default_path, "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)")
        if file_path:
            try:
                img_format = None; ext = os.path.splitext(file_path)[1].lower()
                if selected_filter.startswith("PNG") or ext == ".png": img_format = "PNG"
                elif selected_filter.startswith("JPEG") or ext in [".jpg", ".jpeg"]: img_format = "JPG"
                elif selected_filter.startswith("BMP") or ext == ".bmp": img_format = "BMP"
                success = pixmap.save(file_path, format=img_format)
                if success: logger.info(f"Snapshot saved: {file_path}"); self.notifications.show_message(f"Snapshot saved: {os.path.basename(file_path)}", level="success")
                else: logger.error(f"Failed to save snapshot to {file_path}."); self.notifications.show_message("Failed to save snapshot.", level="error")
            except Exception as e: logger.error(f"Exception saving snapshot: {e}", exc_info=True); self.notifications.show_message(f"Error saving snapshot: {e}", level="error")

    @pyqtSlot(str, float, float, float)
    def start_ptz(self, camera_name: str, pan: float, tilt: float, zoom: float):
         """Sends a continuous PTZ move command."""
         logger.debug(f"PTZ Start: {camera_name}, P={pan}, T={tilt}, Z={zoom}")
         if camera := self.cameras.get(camera_name): camera.move_ptz(pan, tilt, zoom)
         else: logger.warning(f"PTZ Start ignored: Camera '{camera_name}' not found.")

    @pyqtSlot(str)
    def stop_ptz(self, camera_name: str):
        """Sends a PTZ stop command."""
        logger.debug(f"PTZ Stop: {camera_name}")
        if camera := self.cameras.get(camera_name): camera.stop_ptz()
        else: logger.warning(f"PTZ Stop ignored: Camera '{camera_name}' not found.")

    # ==================== Map View Tab Methods ====================
    def create_map_view_tab(self):
        """Creates the Map View tab and its components."""
        map_tab = QWidget(); layout = QVBoxLayout(map_tab); layout.setContentsMargins(0,0,0,0); layout.setSpacing(0)
        toolbar = QToolBar("Map Tools"); toolbar.setIconSize(QSize(18, 18)); toolbar.setMovable(False); layout.addWidget(toolbar)
        load_map_action = QAction(QIcon.fromTheme("document-open", self._create_default_icon()), "Load Map Image...", self); load_map_action.setToolTip("Load background image"); load_map_action.triggered.connect(self.select_and_load_map_image); toolbar.addAction(load_map_action)
        toolbar.addSeparator()
        self.map_edit_mode_action = QAction(QIcon.fromTheme("document-edit", self._create_default_icon()), "Edit Layout", self); self.map_edit_mode_action.setToolTip("Toggle moving camera markers"); self.map_edit_mode_action.setCheckable(True); self.map_edit_mode_action.setChecked(self.map_edit_mode); self.map_edit_mode_action.triggered.connect(self.toggle_map_edit_mode); toolbar.addAction(self.map_edit_mode_action)
        toolbar.addSeparator()
        zoom_in = QAction(QIcon.fromTheme("zoom-in", self._create_default_icon()), "Zoom In", self); zoom_in.triggered.connect(lambda: self.map_view.scale(1.2, 1.2) if self.map_view else None); zoom_in.setShortcut("Ctrl++"); toolbar.addAction(zoom_in)
        zoom_out = QAction(QIcon.fromTheme("zoom-out", self._create_default_icon()), "Zoom Out", self); zoom_out.triggered.connect(lambda: self.map_view.scale(1/1.2, 1/1.2) if self.map_view else None); zoom_out.setShortcut("Ctrl+-"); toolbar.addAction(zoom_out)
        zoom_fit = QAction(QIcon.fromTheme("zoom-fit-best", self._create_default_icon()), "Fit View", self); zoom_fit.triggered.connect(self.fit_map_to_view); zoom_fit.setShortcut("Ctrl+0"); toolbar.addAction(zoom_fit)
        pan_mode = QAction(QIcon.fromTheme("transform-move", self._create_default_icon()), "Pan Mode", self); pan_mode.setToolTip("Click and drag map"); pan_mode.setCheckable(True); pan_mode.setChecked(True); pan_mode.triggered.connect(lambda: self.map_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag) if self.map_view else None)
        mode_group = QActionGroup(self); mode_group.addAction(pan_mode); mode_group.setExclusive(True); toolbar.addSeparator(); toolbar.addAction(pan_mode)

        self.map_scene = QGraphicsScene(self)
        self.map_view = QGraphicsView(self.map_scene); self.map_view.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.map_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag); self.map_view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse); self.map_view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        view_bg = self.palette().color(QPalette.ColorRole.AlternateBase); self.map_view.setBackgroundBrush(view_bg)
        scene_bg = self.palette().color(QPalette.ColorRole.Base); self.map_scene.setBackgroundBrush(scene_bg)
        self.map_view.setStyleSheet("QGraphicsView { border: 1px solid #444; }")
        layout.addWidget(self.map_view)
        self.load_map_image()
        self.tabs.insertTab(1, map_tab, QIcon.fromTheme("applications-geomap", self._create_default_icon("app")), "Map View")
        logger.debug("Map View tab created.")

    def select_and_load_map_image(self):
        """Opens file dialog to select map image and loads it."""
        current_path = self.app_config['map_view'].get('image_path')
        start_dir = os.path.dirname(current_path) if current_path and os.path.exists(os.path.dirname(current_path)) else ""
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Map Background Image", start_dir, "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)")
        if filepath and self.app_config['map_view'].get('image_path') != filepath:
             logger.info(f"New map image selected: {filepath}")
             self.app_config['map_view']['image_path'] = filepath; self.load_map_image(); self.mark_settings_dirty()
             reply = QMessageBox.question(self, "Save Configuration?", f"Map image changed to '{os.path.basename(filepath)}'.\nSave config now?", QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
             if reply == QMessageBox.StandardButton.Save: self.save_config()

    def load_map_image(self):
        """Loads the map background image specified in config into the scene."""
        if not self.map_scene or not self.map_view: return
        map_path = self.app_config['map_view'].get('image_path')
        logger.debug(f"Attempting to load map image from: {map_path}")
        if self.map_background_item:
            if self.map_background_item in self.map_scene.items(): self.map_scene.removeItem(self.map_background_item)
            self.map_background_item = None; logger.debug("Removed existing map background.")
        if map_path and os.path.exists(map_path):
            try:
                pixmap = QPixmap(map_path)
                if pixmap.isNull(): raise ValueError(f"Loaded pixmap is null (unsupported format or corrupt?): {map_path}")
                self.map_background_item = QGraphicsPixmapItem(pixmap); self.map_background_item.setZValue(-10)
                self.map_scene.addItem(self.map_background_item); self.map_scene.setSceneRect(self.map_background_item.boundingRect())
                self.fit_map_to_view(); logger.info(f"Loaded map image: {map_path}")
                if hasattr(self, 'status_bar'): self.status_bar.showMessage(f"Map loaded: {os.path.basename(map_path)}", 5000)
            except Exception as e:
                logger.error(f"Failed to load map image '{map_path}': {e}", exc_info=True)
                if hasattr(self, 'notifications'): self.notifications.show_message(f"Error loading map: {e}", level="error")
                if self.map_scene.sceneRect().isEmpty(): self.map_scene.setSceneRect(QRectF(0,0,800,600))
                self.fit_map_to_view()
        else:
            if map_path: logger.warning(f"Map image file not found: {map_path}"); self.notifications.show_message(f"Map image not found: {os.path.basename(map_path)}", level="warning") if hasattr(self, 'notifications') else None
            else: logger.info("No map image configured."); self.status_bar.showMessage("No map image loaded.", 5000) if hasattr(self, 'status_bar') else None
            if self.map_scene.sceneRect().isEmpty() and not self.map_scene.items(): self.map_scene.setSceneRect(QRectF(0,0,800,600))
            self.fit_map_to_view()

    def fit_map_to_view(self):
        """Fits the current scene contents (map and/or markers) into the view."""
        if not self.map_view or not self.map_scene: return
        rect_to_fit = QRectF()
        if self.map_background_item: rect_to_fit = self.map_background_item.boundingRect()
        else: rect_to_fit = self.map_scene.itemsBoundingRect()
        if not rect_to_fit.isValid() or rect_to_fit.isEmpty(): rect_to_fit = self.map_scene.sceneRect()
        if rect_to_fit.isValid() and not rect_to_fit.isEmpty() and rect_to_fit.width() > 0 and rect_to_fit.height() > 0:
             margin_x = rect_to_fit.width() * 0.05; margin_y = rect_to_fit.height() * 0.05
             rect_with_margin = rect_to_fit.adjusted(-margin_x, -margin_y, margin_x, margin_y)
             self.map_view.fitInView(rect_with_margin, Qt.AspectRatioMode.KeepAspectRatio); logger.debug(f"Fitted map view to rect: {rect_to_fit}")
        else: logger.debug("fit_map_to_view skipped: Scene/items rect empty/invalid.")

    def update_map_markers(self):
        """Synchronizes map markers with the current camera configuration and positions."""
        if not self.map_scene or not self._default_camera_icon: return
        logger.debug("Updating map markers...")
        current_config_names = {cam.get('name') for cam in self.app_config.get('cameras', []) if cam.get('name')}
        camera_positions = self.app_config['map_view'].get('camera_positions', {})
        existing_marker_names = set(self.map_markers.keys())
        names_to_remove = existing_marker_names - current_config_names
        for name in names_to_remove:
            marker = self.map_markers.pop(name, None)
            if marker and marker in self.map_scene.items(): self.map_scene.removeItem(marker)
            logger.debug(f"Removed map marker for deleted camera: {name}")
        for i, config in enumerate(self.app_config.get('cameras', [])):
            cam_name = config.get('name'); marker = self.map_markers.get(cam_name)
            pos_data = camera_positions.get(cam_name); position = None
            if isinstance(pos_data,dict) and 'x' in pos_data and 'y' in pos_data:
                try: position = QPointF(float(pos_data['x']), float(pos_data['y']))
                except: logger.warning(f"Invalid position data for '{cam_name}'."); position = None
            if marker: # Exists
                if position is not None and marker.pos() != position: logger.debug(f"Updating pos for '{cam_name}'"); marker.setPos(position)
                marker.setEditMode(self.map_edit_mode)
                if marker not in self.map_scene.items(): self.map_scene.addItem(marker)
            elif position is not None: # New with position
                try:
                    logger.debug(f"Creating new marker for '{cam_name}' at {position}")
                    marker = CameraMarkerItem(cam_name, self._default_camera_icon.copy(), position)
                    marker.setEditMode(self.map_edit_mode)
                    marker.markerClicked.connect(self.on_marker_clicked) # Direct connect (name passed by sender)
                    marker.markerMoved.connect(self.on_marker_moved)     # Direct connect (name, pos passed by sender)
                    self.map_scene.addItem(marker); self.map_markers[cam_name] = marker
                except Exception as e: logger.error(f"Failed to create marker for {cam_name}: {e}", exc_info=True)
            # else: New without position - added only when entering edit mode
        logger.debug(f"Map markers updated. Count: {len(self.map_markers)}")

    @pyqtSlot(bool)
    def toggle_map_edit_mode(self, checked: bool):
        """Toggles the map editing mode, enabling/disabling marker movement."""
        if self.map_edit_mode == checked: return
        self.map_edit_mode = checked
        logger.info(f"Map Edit Mode Toggled: {'ON' if checked else 'OFF'}")
        self.status_bar.showMessage(f"Map Edit Mode: {'ENABLED' if checked else 'DISABLED'}", 3000) if hasattr(self, 'status_bar') else None
        for marker in self.map_markers.values(): marker.setEditMode(self.map_edit_mode)
        if checked: self._add_markers_for_unplaced_cameras()
        elif not checked and self._settings_dirty:
             reply = QMessageBox.question(self, "Save Map Layout?", "Save unsaved camera positions?", QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
             if reply == QMessageBox.StandardButton.Save: self.save_config()
             elif reply == QMessageBox.StandardButton.Cancel: self.map_edit_mode = True; self.map_edit_mode_action.setChecked(True); logger.info("Map Edit toggle cancelled.")

    def _add_markers_for_unplaced_cameras(self):
        """Adds markers for cameras in config that don't have saved positions."""
        logger.debug("Checking for unplaced cameras...")
        if not self.map_scene or not self._default_camera_icon: return
        current_config_names = {cam.get('name') for cam in self.app_config.get('cameras', []) if cam.get('name')}
        unplaced_names = current_config_names - set(self.map_markers.keys())
        if not unplaced_names: logger.debug("No unplaced cameras."); return
        logger.info(f"Found {len(unplaced_names)} unplaced cameras: {unplaced_names}. Adding markers...")
        scene_rect = self.map_scene.sceneRect(); center_pos = scene_rect.center() if scene_rect.isValid() else QPointF(50,50)
        radius = 40; angle_step = 360.0 / len(unplaced_names) if len(unplaced_names) > 0 else 0
        for i, cam_name in enumerate(unplaced_names):
            angle = i * angle_step * (3.14159 / 180.0)
            initial_pos = center_pos + QPointF(radius * np.cos(angle), radius * np.sin(angle))
            try: # Constrain position
                temp_marker = CameraMarkerItem(cam_name, self._default_camera_icon, QPointF(0,0)); temp_marker.setEditMode(True)
                temp_marker.scene = lambda: self.map_scene # Mock scene access
                constrained_pos = temp_marker.itemChange(QGraphicsItem.GraphicsItemChange.ItemPositionChange, initial_pos); del temp_marker
            except: constrained_pos = initial_pos
            logger.debug(f"Placing new marker for '{cam_name}' near {constrained_pos}")
            try:
                marker = CameraMarkerItem(cam_name, self._default_camera_icon.copy(), constrained_pos); marker.setEditMode(True)
                marker.markerClicked.connect(self.on_marker_clicked); marker.markerMoved.connect(self.on_marker_moved)
                self.map_scene.addItem(marker); self.map_markers[cam_name] = marker
                self.on_marker_moved(cam_name, constrained_pos) # Add to config dict immediately
            except Exception as e: logger.error(f"Failed to create initial marker for '{cam_name}': {e}", exc_info=True)
        self.mark_settings_dirty() # Mark dirty as positions were added

    @pyqtSlot(str, QPointF)
    def on_marker_moved(self, camera_name: str, new_pos: QPointF):
         """Handles the signal when a marker has finished moving."""
         if camera_name in self.map_markers:
              logger.info(f"Marker '{camera_name}' finished moving to ({new_pos.x():.1f}, {new_pos.y():.1f})")
              if 'map_view' not in self.app_config: self.app_config['map_view'] = {}
              if 'camera_positions' not in self.app_config['map_view']: self.app_config['map_view']['camera_positions'] = {}
              rounded_pos = {'x': round(new_pos.x(), 2), 'y': round(new_pos.y(), 2)}
              if self.app_config['map_view']['camera_positions'].get(camera_name) != rounded_pos:
                    self.app_config['map_view']['camera_positions'][camera_name] = rounded_pos
                    self.mark_settings_dirty()
         else: logger.warning(f"Received markerMoved for unknown marker: {camera_name}")

    @pyqtSlot(str)
    def on_marker_clicked(self, camera_name: str):
        """Handles clicks on camera markers (only when not in edit mode)."""
        if self.map_edit_mode: logger.debug(f"Marker click ignored for '{camera_name}': Edit Mode ON."); return
        logger.info(f"Map marker clicked: '{camera_name}'. Switching to monitor tab...")
        monitor_tab_index = -1
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Monitoring": monitor_tab_index = i; break
        if monitor_tab_index == -1: logger.warning("Monitor tab not found."); return
        self.tabs.setCurrentIndex(monitor_tab_index)
        if camera_name in self.camera_group_boxes:
            group_box = self.camera_group_boxes[camera_name]
            monitor_tab_widget = self.tabs.widget(monitor_tab_index)
            scroll_area = monitor_tab_widget.findChild(QScrollArea)
            if scroll_area: scroll_area.ensureWidgetVisible(group_box, yMargin=50); QApplication.processEvents()
            else: logger.warning(f"ScrollArea not found on monitor tab.")
            self.highlight_widget(group_box, duration_ms=1200, color=QColor(42, 130, 218))
        else: logger.warning(f"GroupBox not found for '{camera_name}'.")

    def highlight_widget(self, widget: QWidget, duration_ms: int = 1500, color: QColor = QColor(Qt.GlobalColor.yellow)):
        """Applies a temporary visual highlight to a widget using QVariantAnimation."""
        if not widget: return
        anim_prop_name = b"_highlight_anim_color"; original_stylesheet = widget.styleSheet()
        if existing_anim := widget.property(anim_prop_name):
             if isinstance(existing_anim, QVariantAnimation): existing_anim.stop(); widget.setStyleSheet(original_stylesheet)
        start_color = color; base_border_color = QColor("#666")
        try:
             style_parts = original_stylesheet.split('border:');
             if len(style_parts) > 1: border_part = style_parts[1].split(';')[0].strip(); color_part = border_part.split()[-1]; temp_color = QColor(color_part);
             if temp_color.isValid(): base_border_color = temp_color
        except: pass
        animation = QVariantAnimation(widget); widget.setProperty(anim_prop_name, animation); animation.setDuration(duration_ms)
        animation.setStartValue(start_color); animation.setEndValue(base_border_color); animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        def update_border_color(current_color):
             border_width = "1px"; style_prefix = "QGroupBox { "; style_suffix = " border-radius: 5px; margin-top: 0.5em; padding: 0.5em 0.3em 0.3em 0.3em; }"
             if 'font-weight: bold;' in original_stylesheet: style_prefix += "font-weight: bold; "
             try: widget.setStyleSheet(f"{style_prefix} border: {border_width} solid {current_color.name()}; {style_suffix}")
             except Exception as e: logger.error(f"Error setting highlight style: {e}"); animation.stop() if animation else None
        animation.valueChanged.connect(update_border_color)
        animation.finished.connect(lambda w=widget, style=original_stylesheet, prop_name=anim_prop_name: self._on_highlight_finished(w, style, prop_name))
        animation.start(QVariantAnimation.DeletionPolicy.DeleteWhenStopped)

    def _on_highlight_finished(self, widget, original_stylesheet, prop_name):
         """Restores original style and cleans up animation property."""
         if widget: widget.setStyleSheet(original_stylesheet); widget.setProperty(prop_name, None)

    # ==================== Settings Tab Methods ====================
    def create_settings_tab(self):
        """Creates the Settings tab UI for configuration."""
        settings_tab=QWidget(); layout=QVBoxLayout(settings_tab); layout.setSpacing(15); layout.setContentsMargins(10,10,10,10)
        cam_group = QGroupBox("Camera Configuration"); cam_group.setToolTip("Add, edit, or remove camera sources."); cam_layout = QHBoxLayout(cam_group); cam_layout.setSpacing(10)
        self.camera_list_widget = QListWidget(); self.camera_list_widget.setToolTip("Double-click to edit."); self.camera_list_widget.itemDoubleClicked.connect(self.edit_camera_config); self.camera_list_widget.setAlternatingRowColors(True); cam_layout.addWidget(self.camera_list_widget, 1)
        cam_buttons_layout = QVBoxLayout(); cam_buttons_layout.setSpacing(8)
        add_btn = QPushButton(QIcon.fromTheme("list-add", self._create_default_icon()), " Add Camera..."); add_btn.clicked.connect(self.add_camera_config); cam_buttons_layout.addWidget(add_btn)
        edit_btn = QPushButton(QIcon.fromTheme("document-edit", self._create_default_icon()), " Edit Selected..."); edit_btn.clicked.connect(self.edit_camera_config); cam_buttons_layout.addWidget(edit_btn)
        remove_btn = QPushButton(QIcon.fromTheme("list-remove", self._create_default_icon()), " Remove Selected"); remove_btn.clicked.connect(self.remove_camera_config); cam_buttons_layout.addWidget(remove_btn)
        cam_buttons_layout.addStretch(); cam_layout.addLayout(cam_buttons_layout); layout.addWidget(cam_group)
        siem_group = QGroupBox("SIEM (Splunk) Integration Settings"); siem_group.setToolTip("Configure Splunk connection."); siem_layout = QFormLayout(siem_group)
        siem_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow); siem_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight); siem_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows); siem_layout.setSpacing(10)
        self.siem_url_input = QLineEdit(); self.siem_url_input.setPlaceholderText("e.g., https://your-splunk:8089"); self.siem_token_input = QLineEdit(); self.siem_token_input.setEchoMode(QLineEdit.EchoMode.Password); self.siem_token_input.setPlaceholderText("HEC Token or Bearer Token"); self.siem_query_input = QTextEdit(); self.siem_query_input.setFixedHeight(80); self.siem_query_input.setPlaceholderText("e.g., search index=main earliest=-24h | table _time, host, severity"); self.siem_query_input.setAcceptRichText(False); self.siem_auth_combo = QComboBox(); self.siem_auth_combo.addItems(["Bearer", "Splunk"]); self.siem_auth_combo.setToolTip("Authorization type"); self.siem_verify_ssl_check = QCheckBox("Verify SSL Certificate"); self.siem_verify_ssl_check.setToolTip("Disable for self-signed certs (less secure)"); self.siem_refresh_input = QLineEdit(); self.siem_refresh_input.setPlaceholderText("e.g., 10 (0=off)"); self.siem_refresh_input.setToolTip("Auto-refresh interval (minutes).")
        siem_layout.addRow("Splunk API URL:", self.siem_url_input); siem_layout.addRow("Auth Type:", self.siem_auth_combo); siem_layout.addRow("API Token:", self.siem_token_input); siem_layout.addRow("Search Query:", self.siem_query_input); siem_layout.addRow(self.siem_verify_ssl_check); siem_layout.addRow("Refresh Interval (min):", self.siem_refresh_input)
        layout.addWidget(siem_group); layout.addStretch()
        apply_btn = QPushButton(QIcon.fromTheme("document-save", self._create_default_icon()), " Apply && Save All Settings"); apply_btn.setToolTip("Apply changes and save config."); apply_btn.setFixedHeight(35); apply_btn.clicked.connect(self.apply_and_save_settings); layout.addWidget(apply_btn, 0, Qt.AlignmentFlag.AlignRight)
        self.refresh_settings_ui()
        self.tabs.addTab(settings_tab, QIcon.fromTheme("preferences-system", self._create_default_icon("app")), "Settings")
        logger.debug("Settings tab created.")

    def refresh_settings_ui(self):
         """Populates the settings tab widgets with current app_config values."""
         logger.debug("Refreshing settings UI from app_config...")
         self.camera_list_widget.clear(); current_cam_names = set()
         for config in self.app_config.get('cameras', []):
             name = config.get('name')
             if name: item = QListWidgetItem(f" {name}"); icon_name = "camera-video" if config.get('onvif') else "network-wired"; item.setIcon(QIcon.fromTheme(icon_name, self._create_default_icon("camera"))); item.setData(Qt.ItemDataRole.UserRole, config); self.camera_list_widget.addItem(item); current_cam_names.add(name)
         logger.debug(f"Populated camera list: {self.camera_list_widget.count()} items.")
         siem_config = self.app_config.get('siem', {})
         self.siem_url_input.setText(siem_config.get('api_url', '')); self.siem_token_input.setText(siem_config.get('token', '')); self.siem_query_input.setPlainText(siem_config.get('query', ''))
         auth_type = siem_config.get('auth_header', 'Bearer'); index = self.siem_auth_combo.findText(auth_type, Qt.MatchFlag.MatchFixedString); self.siem_auth_combo.setCurrentIndex(index if index >= 0 else 0)
         self.siem_verify_ssl_check.setChecked(bool(siem_config.get('verify_ssl', False))); self.siem_refresh_input.setText(str(siem_config.get('refresh_interval_min', 15)))
         logger.debug("Populated SIEM fields.")
         map_positions = self.app_config['map_view'].get('camera_positions', {})
         if isinstance(map_positions, dict):
              stale_map_names = set(map_positions.keys()) - current_cam_names
              if stale_map_names: logger.info(f"Pruning stale map positions: {stale_map_names}"); [map_positions.pop(name, None) for name in stale_map_names]
         else: logger.warning("'camera_positions' not a dict. Resetting."); self.app_config['map_view']['camera_positions'] = {}

    def add_camera_config(self):
        """Opens dialog to add a new camera configuration."""
        logger.debug("Add Camera button clicked.")
        dialog = CameraConfigDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if new_config := dialog.get_config():
                 new_name = new_config['name']
                 if new_name in {cfg.get('name') for cfg in self.app_config['cameras']}: QMessageBox.warning(self, "Duplicate Name", f"Camera '{new_name}' already exists."); return
                 logger.info(f"Adding new camera: {new_name}"); self.app_config['cameras'].append(new_config); self.refresh_settings_ui(); self.mark_settings_dirty()
                 self.notifications.show_message(f"Camera '{new_name}' added. Apply & Save to activate.", level="info")

    def edit_camera_config(self):
        """Opens dialog to edit the selected camera configuration."""
        selected_items = self.camera_list_widget.selectedItems(); current_item = self.camera_list_widget.currentItem()
        if not selected_items and not current_item: QMessageBox.information(self, "Edit Camera", "Select camera to edit."); return
        selected_item = selected_items[0] if selected_items else current_item
        current_config = selected_item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(current_config, dict): logger.error(f"Invalid config data for item: {selected_item.text()}."); QMessageBox.critical(self, "Error", "Internal error retrieving config."); return
        original_name = current_config.get('name'); logger.debug(f"Editing camera: {original_name}")
        import copy; dialog = CameraConfigDialog(self, config=copy.deepcopy(current_config))
        if dialog.exec() == QDialog.DialogCode.Accepted:
             if updated_config := dialog.get_config():
                  new_name = updated_config.get('name')
                  if original_name != new_name and new_name in {cfg.get('name') for cfg in self.app_config['cameras'] if cfg.get('name') != original_name}: QMessageBox.warning(self, "Duplicate Name", f"Name '{new_name}' already exists."); return
                  found_index = -1
                  for i, cfg in enumerate(self.app_config['cameras']):
                      if cfg.get('name') == original_name: found_index = i; break
                  if found_index != -1:
                       logger.info(f"Updating config for '{original_name}' (new: '{new_name}')"); self.app_config['cameras'][found_index] = updated_config
                       if original_name != new_name: self._handle_camera_rename_in_map(original_name, new_name)
                       self.refresh_settings_ui(); self.mark_settings_dirty(); self.notifications.show_message(f"Camera '{new_name}' updated. Apply & Save settings.", level="info")
                  else: logger.error(f"Consistency error finding '{original_name}' during edit."); QMessageBox.critical(self, "Error", "Internal error updating camera.")

    def remove_camera_config(self):
        """Removes the selected camera configuration."""
        selected_items = self.camera_list_widget.selectedItems(); current_item = self.camera_list_widget.currentItem()
        if not selected_items and not current_item: QMessageBox.information(self, "Remove Camera", "Select camera to remove."); return
        selected_item = selected_items[0] if selected_items else current_item
        config_data = selected_item.data(Qt.ItemDataRole.UserRole); camera_name = config_data.get('name') if isinstance(config_data, dict) else None
        if not camera_name: logger.error("Failed to get name for removal."); QMessageBox.critical(self, "Error", "Cannot identify camera."); return
        reply = QMessageBox.question(self, "Confirm Removal", f"Remove camera '{camera_name}'?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            logger.info(f"Removing camera: {camera_name}"); initial_len = len(self.app_config['cameras'])
            self.app_config['cameras'] = [cfg for cfg in self.app_config['cameras'] if cfg.get('name') != camera_name]
            if len(self.app_config['cameras']) < initial_len:
                 self._handle_camera_remove_from_map(camera_name); self.refresh_settings_ui(); self.mark_settings_dirty()
                 self.notifications.show_message(f"Camera '{camera_name}' removed. Apply & Save.", level="info")
            else: logger.error(f"Consistency error removing '{camera_name}'."); QMessageBox.warning(self, "Error", "Internal error removing camera.")

    def _handle_camera_rename_in_map(self, old_name: str, new_name: str):
        """Updates the camera name key in the map positions dictionary."""
        if 'map_view' in self.app_config and isinstance(positions := self.app_config['map_view'].get('camera_positions', {}), dict) and old_name in positions:
            positions[new_name] = positions.pop(old_name); logger.debug(f"Updated map key: '{old_name}' -> '{new_name}'."); self.mark_settings_dirty()

    def _handle_camera_remove_from_map(self, camera_name: str):
        """Removes the camera's position from the map positions dictionary."""
        if 'map_view' in self.app_config and isinstance(positions := self.app_config['map_view'].get('camera_positions', {}), dict) and camera_name in positions:
            positions.pop(camera_name); logger.debug(f"Removed map position for '{camera_name}'."); self.mark_settings_dirty()

    def apply_and_save_settings(self):
        """Applies settings from UI, saves config, reinitializes system, and updates UI."""
        logger.info("Apply & Save Settings button clicked...")
        if 'siem' not in self.app_config: self.app_config['siem'] = {}; siem_conf = self.app_config['siem']
        siem_conf['api_url'] = self.siem_url_input.text().strip(); siem_conf['token'] = self.siem_token_input.text()
        siem_conf['query'] = self.siem_query_input.toPlainText().strip(); siem_conf['auth_header'] = self.siem_auth_combo.currentText()
        siem_conf['verify_ssl'] = self.siem_verify_ssl_check.isChecked()
        try: refresh_min = int(refresh_text) if (refresh_text := self.siem_refresh_input.text().strip()) else 0; siem_conf['refresh_interval_min'] = max(0, refresh_min)
        except: current_val = siem_conf.get('refresh_interval_min', 15); logger.warning(f"Invalid SIEM interval. Keeping {current_val}"); self.siem_refresh_input.setText(str(current_val))
        if self.save_config():
             self.notifications.show_message("Applying changes...", level="info", duration=1500); QApplication.processEvents()
             self.stop_all_cameras(); self.init_system(); self.update_siem_timer_interval(); self.recreate_monitor_tab()
             self.notifications.show_message("Settings applied and saved!", level="success", duration=3500)
        else: QMessageBox.critical(self, "Save Error", "Failed to save config. Changes not applied.")

    # ==================== Theme & Misc Methods ====================
    def apply_dark_theme(self):
        """Applies a dark theme palette and stylesheet."""
        logger.debug("Applying dark theme...")
        dp = QPalette(); # Use dp abbreviation
        WINDOW_BG=QColor(53,53,53); WINDOW_TEXT=QColor(230,230,230); BASE=QColor(35,35,35); ALT_BASE=QColor(45,45,45)
        TOOLTIP_BG=QColor(25,25,25); TOOLTIP_TEXT=QColor(230,230,230); TEXT=QColor(220,220,220); BUTTON_BG=QColor(66,66,66)
        BUTTON_TEXT=QColor(230,230,230); BUTTON_DISABLED_TEXT=QColor(127,127,127); BRIGHT_TEXT=QColor(255,80,80)
        HIGHLIGHT=QColor(42,130,218); HIGHLIGHTED_TEXT=QColor(255,255,255); HIGHLIGHT_DISABLED=QColor(80,80,80)
        LINK=QColor(80,160,240); LINK_VISITED=QColor(160,100,220); BORDER_COLOR=QColor(80,80,80)
        dp.setColor(QPalette.ColorRole.Window, WINDOW_BG); dp.setColor(QPalette.ColorRole.WindowText, WINDOW_TEXT); dp.setColor(QPalette.ColorRole.Base, BASE)
        dp.setColor(QPalette.ColorRole.AlternateBase, ALT_BASE); dp.setColor(QPalette.ColorRole.ToolTipBase, TOOLTIP_BG); dp.setColor(QPalette.ColorRole.ToolTipText, TOOLTIP_TEXT)
        dp.setColor(QPalette.ColorRole.Text, TEXT); dp.setColor(QPalette.ColorRole.Button, BUTTON_BG); dp.setColor(QPalette.ColorRole.ButtonText, BUTTON_TEXT)
        dp.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, BUTTON_DISABLED_TEXT); dp.setColor(QPalette.ColorRole.BrightText, BRIGHT_TEXT)
        dp.setColor(QPalette.ColorRole.Highlight, HIGHLIGHT); dp.setColor(QPalette.ColorRole.HighlightedText, HIGHLIGHTED_TEXT)
        dp.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, HIGHLIGHT_DISABLED); dp.setColor(QPalette.ColorRole.Link, LINK); dp.setColor(QPalette.ColorRole.LinkVisited, LINK_VISITED)
        app = QApplication.instance(); app.setPalette(dp) if app else None
        stylesheet=f"""QWidget{{font-size:9pt;}}QMainWindow,QDialog{{background-color:{WINDOW_BG.name()};}}QToolTip{{color:{TOOLTIP_TEXT.name()};background-color:{TOOLTIP_BG.name()};border:1px solid #3b3b3b;padding:5px;border-radius:3px;}}QGroupBox{{font-weight:bold;color:#ddd;border:1px solid {BORDER_COLOR.name()};border-radius:6px;margin-top:0.6em;padding:0.8em 0.5em 0.5em 0.5em;}}QGroupBox::title{{subcontrol-origin:margin;subcontrol-position:top left;padding:0 5px;left:10px;color:#ccc;}}QTabWidget::pane{{border:1px solid {BORDER_COLOR.darker(110).name()};border-radius:3px;margin-top:-1px;background-color:{BASE.name()};}}QTabBar::tab{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #666,stop:1 #555);border:1px solid {BORDER_COLOR.darker(110).name()};border-bottom:none;border-top-left-radius:5px;border-top-right-radius:5px;min-width:10ex;padding:6px 12px;margin-right:2px;color:#ccc;}}QTabBar::tab:selected{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #4a4a4a,stop:1 {BASE.name()});border-color:{BORDER_COLOR.darker(110).name()};color:#fff;font-weight:bold;}}QTabBar::tab:!selected{{margin-top:2px;background:#555;}}QTabBar::tab:!selected:hover{{background:#777;color:#fff;}}QPushButton{{color:{BUTTON_TEXT.name()};background-color:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #666,stop:1 #5a5a5a);border:1px solid {BORDER_COLOR.name()};border-radius:4px;padding:6px 12px;min-width:60px;}}QPushButton:hover{{background-color:#777;border-color:{BORDER_COLOR.lighter(110).name()};}}QPushButton:pressed{{background-color:#505050;}}QPushButton:checked{{background-color:{HIGHLIGHT.name()};border-color:{HIGHLIGHT.darker(120).name()};color:{HIGHLIGHTED_TEXT.name()};}}QPushButton:disabled{{color:{BUTTON_DISABLED_TEXT.name()};background-color:#444;border-color:#555;}}QLineEdit,QComboBox,QAbstractSpinBox{{color:{TEXT.name()};background-color:{ALT_BASE.name()};border:1px solid {BORDER_COLOR.name()};border-radius:4px;padding:4px 6px;}}QTextEdit{{color:{TEXT.name()};background-color:{ALT_BASE.name()};border:1px solid {BORDER_COLOR.name()};border-radius:4px;padding:5px;}}QComboBox::drop-down{{border:none;subcontrol-origin:padding;subcontrol-position:top right;width:18px;}}QListWidget{{color:{TEXT.name()};background-color:{BASE.name()};border:1px solid {BORDER_COLOR.name()};border-radius:4px;padding:2px;alternate-background-color:{ALT_BASE.name()};}}QListWidget::item{{padding:4px 0px;}}QListWidget::item:selected{{background-color:{HIGHLIGHT.name()};color:{HIGHLIGHTED_TEXT.name()};border:none;}}QListWidget::item:selected:!active{{background-color:{HIGHLIGHT.darker(120).name()};}}QCheckBox{{spacing:8px;}}QCheckBox::indicator{{width:16px;height:16px;border:1px solid {BORDER_COLOR.name()};border-radius:4px;background-color:{ALT_BASE.name()};}}QCheckBox::indicator:checked{{background-color:{HIGHLIGHT.name()};border-color:{HIGHLIGHT.darker(120).name()};}}QCheckBox::indicator:disabled{{background-color:#444;border-color:#555;}}QToolBar{{background-color:{WINDOW_BG.darker(110).name()};border:none;padding:3px;spacing:4px;}}QToolButton{{background-color:transparent;border:none;padding:4px;border-radius:4px;color:{BUTTON_TEXT.name()};}}QToolButton:hover{{background-color:{BUTTON_BG.lighter(120).name()};}}QToolButton:pressed{{background-color:{BUTTON_BG.name()};}}QToolButton:checked{{background-color:{HIGHLIGHT.name()};border:1px solid {HIGHLIGHT.darker(120).name()};color:{HIGHLIGHTED_TEXT.name()};}}QStatusBar{{color:#bbb;}}QStatusBar::item{{border:none;}}QGraphicsView{{border:1px solid {BORDER_COLOR.name()};border-radius:3px;}}QScrollArea{{border:none;}}QScrollBar:vertical{{border:1px solid {BORDER_COLOR.name()};background:{BASE.name()};width:12px;margin:0px;}}QScrollBar::handle:vertical{{background:{BUTTON_BG.name()};min-height:20px;border-radius:5px;}}QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0px;background:none;}}QScrollBar:horizontal{{border:1px solid {BORDER_COLOR.name()};background:{BASE.name()};height:12px;margin:0px;}}QScrollBar::handle:horizontal{{background:{BUTTON_BG.name()};min-width:20px;border-radius:5px;}}QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal{{width:0px;background:none;}}"""
        if app: app.setStyleSheet(stylesheet)
        logger.debug("Dark theme applied.")

    def show_about(self):
        """Displays the About dialog box."""
        try: py_ver = platform.python_version()
        except: py_ver = "N/A"
        try: qt_ver = Qt.PYQT_VERSION_STR
        except: qt_ver = "N/A"
        try: cv_ver = cv2.__version__
        except: cv_ver = "N/A"
        try: import importlib.metadata; onvif_ver = importlib.metadata.version('onvif_zeep')
        except: onvif_ver = "N/A"
        app_version = "1.3.0"
        about_text = f"""<h2>Security Monitor Pro</h2><p>Version: {app_version}</p><p>Comprehensive security monitoring.</p><hr><p><b>Runtime Info:</b></p><ul><li>Python: {py_ver}</li><li>PyQt: {qt_ver}</li><li>OpenCV: {cv_ver}</li><li>ONVIF-Zeep: {onvif_ver}</li><li>Platform: {platform.system()} ({platform.release()})</li></ul><hr><p><b>Features:</b></p><ul><li>RTSP & ONVIF (PTZ) Support</li><li>Splunk SIEM Integration</li><li>OpenCV Motion Detection</li><li>Visual Map View</li><li>Dynamic Camera Config</li><li>Dark Mode UI</li><li>Snapshots</li><li>Auto Dependency Install</li></ul><p style='font-size:8pt;color:#aaa;'><i>Note: Functionality depends on camera firmware & network.</i></p>"""
        QMessageBox.about(self, f"About Security Monitor Pro v{app_version}", about_text)

    def resizeEvent(self, event: 'QResizeEvent'):
        """Handles window resize events."""
        super().resizeEvent(event)
        if hasattr(self, 'notifications') and self.notifications and self.notifications.isVisible():
            parent_width = self.central_widget.width(); notif_width = self.notifications.width()
            new_x = (parent_width - notif_width) // 2; current_y = self.notifications.geometry().y()
            if current_y < 10: current_y = 20
            self.notifications.move(new_x, current_y)

    def check_unsaved_changes(self, action_desc: str = "perform this action") -> bool:
         """Checks for unsaved changes and prompts user. Returns True if safe to proceed."""
         if not self._settings_dirty: return True
         reply = QMessageBox.question(self, "Unsaved Changes", f"Save unsaved changes before you {action_desc}?", QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
         if reply == QMessageBox.StandardButton.Save: return self.save_config()
         elif reply == QMessageBox.StandardButton.Discard: logger.info("Discarding unsaved changes."); self._settings_dirty = False; self.update_window_title(); return True
         else: logger.info(f"Action '{action_desc}' cancelled."); return False

    def closeEvent(self, event: 'QCloseEvent'):
        """Handles the main window close event."""
        logger.info("Close event triggered. Initiating shutdown...");
        if not self.check_unsaved_changes("exit the application"): event.ignore(); logger.info("Application close cancelled."); return
        if hasattr(self, 'status_bar'): self.status_bar.showMessage("Shutting down..."); QApplication.processEvents()
        if hasattr(self, 'siem_refresh_timer'): self.siem_refresh_timer.stop(); logger.debug("SIEM timer stopped.")
        self.stop_all_cameras() # Stop threads gracefully
        logger.info("Shutdown sequence complete. Exiting application.")
        event.accept()

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # --- Platform-specific setup / Environment hints ---
    # High DPI scaling is generally handled well by Qt6, but explicit settings can help sometimes.
    # On Windows: Enable High DPI support (Environment variable often works)
    if platform.system() == "Windows":
        # Setting environment variables might be needed *before* QApplication is created
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
        # os.environ["QT_SCALE_FACTOR"] = "1.0" # Example: Force scale factor if needed
        # QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough) # Experiment if scaling issues persist

    # On Linux (Wayland/X11): Hint preference if needed
    # os.environ["QT_QPA_PLATFORM"] = "wayland;xcb" # Prioritize Wayland

    # --- Application Setup ---
    QApplication.setApplicationName("SecurityMonitorPro")
    QApplication.setOrganizationName("UserProject") # Optional, used for settings paths etc.

    # REMOVED/COMMENTED OUT the problematic line:
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling) # <- REMOVE THIS LINE

    app = QApplication(sys.argv)

    # Set 'Fusion' style for a consistent look across platforms (often good with dark themes)
    app.setStyle('Fusion')

    # --- Main Window Initialization and Run ---
    window = None # Initialize reference
    try:
        logger.info("Creating main application window...")
        window = SecurityMonitorApp() # Instantiate the main application class
        window.show() # Display the window
        logger.info("Application window shown. Starting event loop.")
        exit_code = app.exec() # Start the Qt event loop
        logger.info(f"Application event loop finished. Exiting with code {exit_code}.")
        sys.exit(exit_code)

    except Exception as e:
         # Catch critical errors during initialization or runtime unhandled exceptions
         logger.critical(f"FATAL ERROR: Unhandled exception during application lifecycle: {e}", exc_info=True)
         # Use QMessageBox for GUI feedback if app object exists, otherwise print
         if QApplication.instance():
              QMessageBox.critical(None, "Fatal Application Error", f"A critical error occurred:\n{e}\n\nThe application will now exit. Please check logs for details.")
         else:
              print(f"FATAL ERROR: {e}", file=sys.stderr)
         sys.exit(1) # Exit with error code