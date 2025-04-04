import os
import sys
import subprocess
import platform
import time
import datetime
import json
from typing import List, Dict, Optional, Tuple, Any, Union
from enum import Enum
import logging
import asyncio # Retained for potential future use
import threading
import copy # For deep copying configs in edit dialogs

# ==================== LOGGING SETUP ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    print("--- Checking Dependencies ---")
    packages_to_install = []
    for pkg, import_name in required.items():
        try:
            if '.' in import_name:
                base_module = import_name.split('.')[0]
                __import__(base_module)
            else:
                __import__(import_name)
        except ImportError:
            print(f" [ ] {pkg} not found.")
            packages_to_install.append(pkg)

    if not packages_to_install:
        print("--- All dependencies are satisfied ---")
        return

    print(f"--- Installing missing packages: {', '.join(packages_to_install)} ---")
    for pkg in packages_to_install:
            logger.info(f"⚙️ Package '{pkg}' not found. Installing...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "--upgrade", pkg, "--disable-pip-version-check", "--no-cache-dir"],
                )
                logger.info(f"✅ Successfully installed {pkg}.")
                installed_something = True
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install {pkg}. Pip exited with status {e.returncode}.")
                print(f"\n❌ ERROR: Failed to install {pkg}.")
                print(f"   Please try installing it manually using:")
                print(f"   '{sys.executable} -m pip install {pkg}'")
                print(f"   Then restart the application.")
            except Exception as e:
                logger.error(f"❌ An unexpected error occurred during installation of {pkg}. Error: {e}", exc_info=True)
                print(f"\n❌ ERROR: An unexpected error occurred installing {pkg}: {e}")
                sys.exit(1)
    print("--- Dependency check complete ---")


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
        QGraphicsDropShadowEffect, QCheckBox, QScrollArea, QMenu, QTableWidget,
        QTableWidgetItem, QAbstractItemView, QHeaderView, QAbstractSpinBox
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
        Fault = TransportError = XMLSyntaxError = Exception # Define for type hinting even if unavailable
        logger.warning("onvif-zeep library not found or failed to import. ONVIF functionality will be disabled.")

    from requests.exceptions import ConnectionError as RequestsConnectionError, Timeout, RequestException
    from PIL import Image

except ImportError as e:
    logger.critical(f"❌ Critical import failed after installation attempt: {e}. Please ensure all dependencies were installed correctly.", exc_info=True)
    try:
        if not QApplication.instance(): app = QApplication([]) # Minimal app instance for message box
        QMessageBox.critical(None, "Import Error", f"Failed to import a required library: {e}\n\nPlease check installation and restart.")
    except:
         print(f"❌ Failed to import a required library: {e}. Exiting.", file=sys.stderr)
    sys.exit(1)


# ==================== SYSTEM COMPONENTS ====================

# --- SecurityCamera Class ---
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
        self.connection_retry_delay: int = 15 # seconds
        self.last_error: Optional[str] = None
        logger.debug(f"Initializing camera object: {self.name} (ONVIF: {self.is_onvif})")

    def _set_error(self, message: str, log_level=logging.ERROR):
        if log_level >= logging.ERROR: logger.error(f"Camera Error ({self.name}): {message}")
        elif log_level == logging.WARNING: logger.warning(f"Camera Warning ({self.name}): {message}")
        else: logger.info(f"Camera Info ({self.name}): {message}")
        self.last_error = message
        if log_level >= logging.ERROR:
            if self.is_connected: logger.info(f"Camera '{self.name}' state changed to DISCONNECTED due to error.")
            self.is_connected = False
            self.release_capture()
            self.ptz = None; self.media_profile = None

    def release_capture(self):
        if self.cap is not None:
            logger.debug(f"Releasing video capture for {self.name}")
            try: self.cap.release()
            except Exception as e: logger.error(f"Exception releasing capture for {self.name}: {e}")
            self.cap = None

    def connect(self) -> bool:
        if self.is_connecting: return False
        if self.is_connected: return True
        current_time = time.time()
        if current_time - self.last_connection_attempt_time < self.connection_retry_delay: return False
        self.is_connecting = True; self.last_connection_attempt_time = current_time; logger.info(f"Attempting connection: {self.name}"); self.last_error = None; self.release_capture(); self.ptz = None
        stream_uri = self.url # Default/fallback
        try:
            # --- ONVIF ---
            if self.is_onvif:
                if not ONVIFCamera: self._set_error("ONVIF lib unavailable."); self.is_connecting = False; return False
                if not self.host or not self.port: self._set_error("ONVIF Host/Port missing."); self.is_connecting = False; return False
                logger.info(f"Connecting ONVIF: {self.host}:{self.port}")
                try: # Find WSDL
                    import onvif; wsdl_dir=os.path.join(os.path.dirname(onvif.__file__),'wsdl')
                    if not os.path.isdir(wsdl_dir):
                         wsdl_alt=os.path.join(os.path.dirname(sys.modules['onvif'].__file__),'wsdl')
                         if os.path.isdir(wsdl_alt): wsdl_dir=wsdl_alt
                         else: raise FileNotFoundError("ONVIF WSDL dir not found.")
                    logger.debug(f"Using ONVIF WSDL: {wsdl_dir}")
                except Exception as e: self._set_error(f"ONVIF WSDL lookup fail: {e}"); self.is_connecting=False; return False

                try: # Connect ONVIF & Get Stream URI & Setup PTZ
                    self.onvif_cam = ONVIFCamera(self.host,self.port,self.user,self.password,wsdl_dir=wsdl_dir,transport_timeout=10)
                    dev_info=self.onvif_cam.devicemgmt.GetDeviceInformation(); logger.info(f"ONVIF ok: {dev_info.Manufacturer} {dev_info.Model}")
                    media=self.onvif_cam.create_media_service(); profiles=media.GetProfiles()
                    if not profiles: self._set_error("No media profiles."); self.is_connecting=False; return False
                    vid_profiles=[p for p in profiles if hasattr(p,'VideoEncoderConfiguration') and p.VideoEncoderConfiguration]; self.media_profile=vid_profiles[0] if vid_profiles else profiles[0]; logger.info(f"Using profile: {self.media_profile.Name}")

                    # Get Stream URI
                    obtained_uri=None
                    for proto in ['TCP','UDP']:
                         try: req=media.create_type('GetStreamUri');req.ProfileToken=self.media_profile.token;req.StreamSetup={'Stream':'RTP-Unicast','Transport':{'Protocol':proto}}; info=media.GetStreamUri(req); obtained_uri=info.Uri; logger.info(f"Got URI ({proto}): {obtained_uri}"); break
                         except Exception as e_uri: logger.warning(f"Fail get {proto} URI ({self.name}): {e_uri}")
                    if not obtained_uri:
                        logger.warning(f"Checking profile attrs for URI ({self.name})...");
                        for attr in ['Uri','uri','MediaUri','RTSPStreamUri','StreamUri']:
                            if hasattr(self.media_profile,attr) and (val:=getattr(self.media_profile,attr)):
                                if isinstance(val,str) and val.lower().startswith("rtsp://"): obtained_uri=val; logger.info(f"Using URI from profile '{attr}'"); break
                                elif isinstance(val,dict) and (u:=val.get('Uri')) and isinstance(u,str) and u.lower().startswith("rtsp://"): obtained_uri=u; logger.info(f"Using URI from profile dict '{attr}'"); break
                        if not obtained_uri: logger.warning(f"No RTSP URI in profile ({self.name}).")

                    # Process URI or Fallback
                    if obtained_uri:
                        stream_uri = obtained_uri
                        if stream_uri.startswith("rtsp://"):
                            uri_parts = stream_uri.split("://"); host_part = uri_parts[1].split("/")[0]
                            if "@" not in host_part:
                                if self.user and self.password: stream_uri=stream_uri.replace("rtsp://",f"rtsp://{self.user}:{self.password}@",1); logger.debug("Injected creds")
                                elif self.user: stream_uri=stream_uri.replace("rtsp://",f"rtsp://{self.user}@",1); logger.debug("Injected user")
                    elif self.url: logger.warning(f"ONVIF URI failed, use config URL: {self.url}"); stream_uri=self.url
                    else: self._set_error("Failed get URI, no fallback URL."); self.is_connecting=False; return False

                    # === CORRECTED PTZ SETUP BLOCK ===
                    try: # Attempt PTZ setup gracefully
                        self.ptz = self.onvif_cam.create_ptz_service()
                        ptz_configs = self.ptz.GetConfigurations()
                        found_token = None
                        # Check linked token
                        if hasattr(self.media_profile, 'PTZConfiguration') and self.media_profile.PTZConfiguration and hasattr(self.media_profile.PTZConfiguration, 'token'):
                            prof_token = self.media_profile.PTZConfiguration.token
                            if any(c.token == prof_token for c in ptz_configs):
                                found_token = prof_token
                                logger.info(f"Using PTZ config from profile: {found_token}")
                        # Fallback to first available
                        if not found_token and ptz_configs:
                            found_token = ptz_configs[0].token
                            logger.warning(f"Using first PTZ config: {found_token}")
                        # Assign token if found
                        if found_token:
                            self.ptz_configuration_token = found_token
                        else:
                            logger.warning(f"No PTZ config found ({self.name}). Disabling PTZ.")
                            self.ptz = None # Disable if no token
                    except (Fault, TransportError, AttributeError, ConnectionRefusedError, TimeoutError) as e_ptz:
                         logger.warning(f"PTZ setup failed ({self.name}), disabling PTZ: {e_ptz}")
                         self.ptz = None # Ensure disabled
                    except Exception as e_ptz_generic:
                         logger.error(f"Unexpected error during PTZ setup ({self.name}): {e_ptz_generic}", exc_info=True)
                         self.ptz = None # Ensure disabled
                    # === END CORRECTED PTZ SETUP BLOCK ===

                # Main ONVIF Exception Handlers
                except (Fault,TransportError,RequestsConnectionError,TimeoutError,ConnectionRefusedError,XMLSyntaxError) as e_onvif: self._set_error(f"ONVIF Conn Fail: {e_onvif}"); self.is_connecting=False; return False
                except AttributeError as e_attr: self._set_error(f"ONVIF Attr Err: {e_attr}"); self.is_connecting=False; return False
                except Exception as e_gen: import traceback; self._set_error(f"ONVIF Gen Err: {e_gen}\n{traceback.format_exc()}"); self.is_connecting=False; return False
            # --- End ONVIF ---

            # --- VideoCapture Init ---
            if not stream_uri: self._set_error(f"No stream URI for {self.name}."); self.is_connecting=False; return False
            log_uri=stream_uri[:stream_uri.find('@')] if '@' in stream_uri else stream_uri[:50]
            logger.info(f"Opening capture ({self.name}): {log_uri}...")
            env_opts={}; orig_env=os.environ.copy(); env_opts['OPENCV_FFMPEG_CAPTURE_OPTIONS']='rtsp_transport;tcp|stimeout;5000000'; os.environ.update(env_opts)
            try:
                self.cap = cv2.VideoCapture(stream_uri, cv2.CAP_FFMPEG)
                if not self.cap or not self.cap.isOpened():
                    logger.warning(f"Capture TCP fail ({self.name}). Try UDP."); os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']='rtsp_transport;udp|stimeout;5000000'
                    self.cap = cv2.VideoCapture(stream_uri, cv2.CAP_FFMPEG)
                    if not self.cap or not self.cap.isOpened():
                        logger.warning(f"Capture UDP fail ({self.name}). Try default."); os.environ.pop('OPENCV_FFMPEG_CAPTURE_OPTIONS', None)
                        self.cap = cv2.VideoCapture(stream_uri, cv2.CAP_FFMPEG)
                        if not self.cap or not self.cap.isOpened():
                            bname="FFMPEG" # Default backend name
                            try:
                                bname = cv2.videoio_registry.getBackendName(cv2.CAP_FFMPEG)
                            except Exception as e_backend:
                                logger.warning(f"Could not get OpenCV backend name: {e_backend}")
                            self._set_error(f"Capture open failed (Backend:{bname}) after retries."); self.is_connecting=False; return False
            finally: # Restore env
                for k in env_opts:
                    if k in orig_env: os.environ[k] = orig_env[k]
                    elif k in os.environ: del os.environ[k]
            # --- Success ---
            self.is_connected = True; self.is_connecting = False; self.last_error = None; logger.info(f"✅ Connected: {self.name}"); self.prev_frame_gray = None; return True
        # General Exception during connect
        except Exception as e: import traceback; self._set_error(f"Connect sequence error: {e}\n{traceback.format_exc()}"); self.is_connecting=False; return False

    def get_frame(self) -> Optional[np.ndarray]:
        if not self.is_connected:
            if not self.connect(): time.sleep(0.5); return None
        if self.cap is None or not self.cap.isOpened(): self._set_error("Capture invalid/closed."); return None
        try: ret,frame=self.cap.read()
        except cv2.error as e: logger.error(f"CV err reading frame ({self.name}): {e}"); self._set_error(f"CV err frame read: {e}"); return None
        except Exception as e: logger.error(f"Unexpected err reading frame ({self.name}): {e}",exc_info=True); self._set_error(f"Unexpected frame read err: {e}"); return None

        if not ret or frame is None: logger.warning(f"Frame read fail ({self.name}). Reconnecting."); self.is_connected=False; self.release_capture(); self.last_connection_attempt_time=0; return None
        return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)


    def detect_motion(self, frame: np.ndarray) -> bool:
        if frame is None or self.motion_threshold<=0: return False
        try:
            gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY); gray=cv2.GaussianBlur(gray,(21,21),0)
            if self.prev_frame_gray is None: self.prev_frame_gray=gray; return False
            delta=cv2.absdiff(self.prev_frame_gray,gray); thresh=cv2.threshold(delta,25,255,cv2.THRESH_BINARY)[1]; thresh=cv2.dilate(thresh,None,iterations=2); self.prev_frame_gray=gray
            cnts,_=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); return any(cv2.contourArea(c)>=self.motion_threshold for c in cnts)
        except cv2.error as e: logger.error(f"CV err motion detect ({self.name}): {e}"); self.prev_frame_gray=None; return False
        except Exception as e: logger.error(f"Unexpected err motion detect ({self.name}): {e}",exc_info=True); self.prev_frame_gray=None; return False

    def _perform_ptz_action(self, action_func, action_name: str="PTZ", *args) -> bool:
        if not self.is_connected: logger.warning(f"PTZ fail ({self.name}): Offline."); return False
        if not self.ptz: return False
        if not self.ptz_configuration_token: logger.warning(f"PTZ fail ({self.name}): No token."); return False
        try: action_func(self.ptz, self.ptz_configuration_token, *args); return True
        except (Fault, TransportError, ConnectionRefusedError, TimeoutError) as e: logger.error(f"ONVIF PTZ {action_name} Fault ({self.name}): {e}"); return False
        except AttributeError as e_attr: logger.error(f"PTZ Attr Err {action_name} ({self.name}): {e_attr}"); return False
        except Exception as e: logger.error(f"PTZ Gen Err {action_name} ({self.name}): {e}", exc_info=True); return False

    def move_ptz(self, pan: float, tilt: float, zoom: float):
        def action(ptz_svc, token, p, t, z):
            req=ptz_svc.create_type('ContinuousMove'); req.ProfileToken=token
            try: Speed=getattr(ptz_svc.zeep_client.get_type('ns0:PTZSpeed'),'__call__'); req.Velocity=Speed(PanTilt={'x':np.clip(p,-1.,1.),'y':np.clip(t,-1.,1.)},Zoom={'x':np.clip(z,-1.,1.)})
            except AttributeError: logger.warning(f"PTZSpeed type fail ({self.name}), try Vector."); Vec=ptz_svc.create_type('PTZVector'); Vec.PanTilt=ptz_svc.create_type('Vector2D',x=np.clip(p,-1.,1.),y=np.clip(t,-1.,1.)); Vec.Zoom=ptz_svc.create_type('Vector1D',x=np.clip(z,-1.,1.)); req.Velocity=Vec
            ptz_svc.ContinuousMove(req)
        self._perform_ptz_action(action, "Move", pan, tilt, zoom)

    def stop_ptz(self):
         def action(ptz_svc, token): req=ptz_svc.create_type('Stop'); req.ProfileToken=token; req.PanTilt=True; req.Zoom=True; ptz_svc.Stop(req)
         self._perform_ptz_action(action, "Stop")

    def release(self): logger.debug(f"Releasing cam: {self.name}"); self.is_connected=False; self.release_capture(); self.ptz=None; self.media_profile=None
    def __del__(self): self.release()

# --- SIEMClient Class ---
class SIEMClient:
    """Handles fetching events from Splunk SIEM API using multiple queries."""
    def __init__(self, config: dict):
        self.config = config; self.session = requests.Session(); self.api_url = config.get('api_url', '').rstrip('/'); self.token = config.get('token', ''); self.queries_config = config.get('queries', []); self.auth_header_type = config.get("auth_header", "Bearer"); self.verify_ssl = config.get("verify_ssl", False)
        self.is_configured = bool(self.api_url and self.token and any(q.get('enabled',False) and q.get('query') for q in self.queries_config))
        if self.is_configured:
            auth_val = f"{'Splunk' if self.auth_header_type.lower()=='splunk' else 'Bearer'} {self.token}"; self.session.headers.update({"Authorization": auth_val})
            if not self.verify_ssl:
                try:
                    import urllib3
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                except ImportError:
                    logger.warning("urllib3 not found, cannot disable InsecureRequestWarning for SIEM.")
                except AttributeError:
                    logger.warning("Could not disable InsecureRequestWarning using urllib3 for SIEM.")
                except Exception as e:
                     logger.warning(f"Error disabling InsecureRequestWarning for SIEM: {e}")
            logger.info(f"SIEM Client ok: {self.api_url}. {len([q for q in self.queries_config if q.get('enabled')])} queries active.")
        else: logger.warning("SIEM Client not configured or no queries enabled.")

    def fetch_alerts(self) -> List[Dict]:
        if not self.is_configured: return []
        all_res=[]; url=f"{self.api_url}/services/search/jobs/export"; hdrs={'Content-Type':'application/x-www-form-urlencoded'}
        for qc in self.queries_config:
            if not qc.get('enabled',False) or not qc.get('query'): continue
            qn=qc.get('name','?'); sq=qc['query'].strip();
            if not sq.lower().startswith('search '): sq=f'search {sq}'
            payload={"search":sq,"output_mode":"json"}; logger.info(f"Fetching SIEM '{qn}'...")
            try: # Outer try for the whole query process
                resp=self.session.post(url,data=payload,headers=hdrs,timeout=45,verify=self.verify_ssl,stream=True); resp.raise_for_status(); q_res=[]; buf=""
                for line in resp.iter_lines(decode_unicode=True):
                    buf+=line;
                    if buf.strip().endswith('}'):
                        try: # Inner try for JSON loading
                            d=json.loads(buf);
                            # Check if loading was successful and yielded a dict
                            if isinstance(d,dict):
                                # --- CORRECTED INDENTATION FOR IF BLOCK ---
                                r=d.get('result',d) # Indented under the outer if
                                if isinstance(r,dict): # Indented under the outer if
                                     r['_query_name']=qn # Further indented under 'if isinstance(r,dict)'
                                     q_res.append(r)   # Further indented under 'if isinstance(r,dict)'
                                # --- END CORRECTION ---
                            # Clear buffer only after successful processing inside the try
                            buf=""
                        except json.JSONDecodeError:
                            # logger.debug(f"Partial JSON received, buffer continues: {buf[:50]}...")
                            pass # Ignore incomplete JSON objects, wait for more data
                if buf.strip(): logger.warning(f"SIEM '{qn}': Buffer remain: {buf[:100]}...")
                logger.info(f"Query '{qn}' -> {len(q_res)} results."); all_res.extend(q_res)
            # Except clauses for the OUTER try
            except Timeout: logger.error(f"SIEM Timeout ('{qn}')")
            except RequestsConnectionError as e: logger.error(f"SIEM Conn Fail ('{qn}'): {e}")
            except RequestException as e:
                logger.error(f"SIEM Req Err ('{qn}'): {e}");
                if e.response is not None: logger.error(f"SIEM Resp: {e.response.status_code}; Body: {e.response.text[:500]}")
            except Exception as e: import traceback; logger.error(f"SIEM Err ('{qn}'): {e}\n{traceback.format_exc()}")
        logger.info(f"Total SIEM results: {len(all_res)}")
        try: all_res.sort(key=lambda x:x.get('_time',''), reverse=True)
        except Exception as e_sort: logger.warning(f"Cannot sort SIEM results: {e_sort}")
        return all_res

# --- SOARClient Class ---
class SOARClient:
    """Handles communication *to* a SOAR platform API to trigger playbooks."""
    def __init__(self, config: dict):
        self.config=config; self.session=requests.Session(); self.api_url=config.get('api_url','').rstrip('/'); self.api_key=config.get('api_key',''); self.auth_hdr_name=config.get('auth_header_name','Authorization'); self.auth_hdr_pfx=config.get('auth_header_prefix','Bearer '); self.verify_ssl=config.get("verify_ssl",True)
        self.is_configured=bool(self.api_url and self.api_key)
        if self.is_configured:
            auth_val=f"{self.auth_hdr_pfx}{self.api_key}"; self.session.headers.update({self.auth_hdr_name:auth_val,"Content-Type":"application/json","Accept":"application/json"})
            if not self.verify_ssl:
                try:
                    import urllib3
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                except ImportError:
                    logger.warning("urllib3 not found, cannot disable InsecureRequestWarning for SOAR.")
                except AttributeError:
                    logger.warning("Could not disable InsecureRequestWarning using urllib3 for SOAR.")
                except Exception as e:
                     logger.warning(f"Error disabling InsecureRequestWarning for SOAR: {e}")
            logger.info(f"SOAR Client ok: {self.api_url}")
        else: logger.warning("SOAR Client not configured.")

    def trigger_playbook(self, playbook_id: Union[str, int], context: Optional[Dict]=None) -> Tuple[bool, str]:
        """Triggers SOAR playbook. **ADAPT endpoint/payload below!**"""
        if not self.is_configured: return False,"SOAR client not configured."
        # --- ADAPT THIS SECTION --- Example: Splunk SOAR (Phantom) ---
        ep=f"{self.api_url}/rest/playbook_run"; payload={"playbook_id":playbook_id,"scope":"new"}
        if context:
             cname=f"SecMon Event {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"; artifacts=[]
             for k,v in context.items():
                 if v: artifacts.append({"name":k,"cef":{k:str(v)},"source_data_identifier":f"secmon_{k}_{int(time.time())}"})
             if artifacts: payload["container"]={"name":cname,"artifacts":artifacts}
        # --- END ADAPTATION ---
        logger.info(f"Triggering SOAR PB ID '{playbook_id}' at {ep}")
        try:
            resp=self.session.post(ep,json=payload,timeout=30,verify=self.verify_ssl); resp.raise_for_status(); data=resp.json()
            run_id=data.get('playbook_run_id') or data.get('id'); success=data.get('success',True)
            if success: msg=f"OK trigger PB {playbook_id}."+(f" Run:{run_id}" if run_id else ""); logger.info(msg); return True,msg
            else: err=data.get('message','SOAR trigger fail.'); logger.error(f"PB trigger fail {playbook_id}: {err}"); return False,f"SOAR Err:{err}"
        except Timeout: logger.error(f"SOAR Timeout: {ep}"); return False,"Timeout."
        except RequestsConnectionError as e: logger.error(f"SOAR Conn Fail: {ep}. Err:{e}"); return False,"Connection fail."
        except RequestException as e:
            logger.error(f"SOAR Req Err: {e}"); msg=f"API Err:{e.response.status_code if e.response else 'N/A'}"
            if e.response is not None:
                try: msg+=f" - {e.response.json().get('message',e.response.text[:100])}"
                except: msg+=f" - {e.response.text[:100]}"
                logger.error(f"SOAR Resp Body: {e.response.text[:500]}");
            return False,msg
        except Exception as e: import traceback; logger.error(f"SOAR Trigger Unexpected Err: {e}\n{traceback.format_exc()}"); return False,f"Unexpected err:{e}"

# --- CameraConfigDialog Class --- (Full implementation)
class CameraConfigDialog(QDialog):
    """Dialog for adding or editing camera configurations."""
    def __init__(self, parent=None, config=None):
        super().__init__(parent); self.setWindowTitle("Camera Configuration"); self.setMinimumWidth(450); self.config=config or {}; ly=QVBoxLayout(self); fl=QFormLayout(); fl.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows); self.name_input=QLineEdit(self.config.get('name','')); self.type_combo=QComboBox(); self.type_combo.addItems(["RTSP URL","ONVIF"]); self.url_input=QLineEdit(self.config.get('url','rtsp://user:pass@host:port/stream')); self.host_input=QLineEdit(self.config.get('host','192.168.1.100')); self.port_input=QLineEdit(str(self.config.get('port',80))); self.user_input=QLineEdit(self.config.get('user','admin')); self.password_input=QLineEdit(self.config.get('password','')); self.password_input.setEchoMode(QLineEdit.EchoMode.Password); self.motion_thresh_input=QLineEdit(str(self.config.get('motion_threshold',500))); fl.addRow("Name*:",self.name_input); fl.addRow("Type:",self.type_combo); self.url_row=QWidget();url_ly=QFormLayout(self.url_row);url_ly.setContentsMargins(0,0,0,0); self.url_lbl=QLabel("RTSP URL*:"); url_ly.addRow(self.url_lbl,self.url_input); fl.addRow(self.url_row); self.onvif_rows=QWidget();onvif_ly=QFormLayout(self.onvif_rows);onvif_ly.setContentsMargins(0,0,0,0); self.host_lbl=QLabel("ONVIF Host*:");self.port_lbl=QLabel("ONVIF Port*:");self.user_lbl=QLabel("ONVIF User:");self.pass_lbl=QLabel("ONVIF Pass:"); onvif_ly.addRow(self.host_lbl,self.host_input); onvif_ly.addRow(self.port_lbl,self.port_input); onvif_ly.addRow(self.user_lbl,self.user_input); onvif_ly.addRow(self.pass_lbl,self.password_input); fl.addRow(self.onvif_rows); fl.addRow("Motion Thresh:",self.motion_thresh_input); self.motion_thresh_input.setToolTip("Contour area(px). 0=off."); ly.addLayout(fl); bb=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel); bb.accepted.connect(self.accept); bb.rejected.connect(self.reject); ly.addWidget(bb); self.type_combo.currentIndexChanged.connect(self.update_fields_visibility); is_onvif=self.config.get('onvif',False); self.type_combo.setCurrentIndex(1 if is_onvif else 0); self.update_fields_visibility()
    def update_fields_visibility(self): is_onvif=(self.type_combo.currentText()=="ONVIF"); self.url_row.setVisible(not is_onvif); self.onvif_rows.setVisible(is_onvif); self.url_lbl.setText("RTSP URL*:" if not is_onvif else "RTSP URL(Fallback):"); self.host_lbl.setText("ONVIF Host*:" if is_onvif else "ONVIF Host:"); self.port_lbl.setText("ONVIF Port*:" if is_onvif else "ONVIF Port:")
    def get_config(self)->Optional[Dict]:
        cfg={}; cfg['name']=self.name_input.text().strip();
        if not cfg['name']:QMessageBox.warning(self,"Input Err","Name required.");return None;
        is_onvif=(self.type_combo.currentText()=="ONVIF"); cfg['onvif']=is_onvif
        if is_onvif:
            cfg['host']=self.host_input.text().strip();
            if not cfg['host']:QMessageBox.warning(self,"Input Err","ONVIF Host required.");return None;
            try: port=int(self.port_input.text().strip()); assert 1<=port<=65535; cfg['port']=port;
            except: QMessageBox.warning(self,"Input Err","Invalid ONVIF Port(1-65535).");return None;
            cfg['user']=self.user_input.text().strip(); cfg['password']=self.password_input.text();
            cfg['url']=self.url_input.text().strip() or None; # Fallback URL is optional for ONVIF
            if cfg['url'] and not cfg['url'].lower().startswith("rtsp://"): QMessageBox.warning(self,"Input Err","Fallback URL must be RTSP.");return None
        else: # RTSP URL mode
            cfg['url']=self.url_input.text().strip();
            if not cfg['url'] or not cfg['url'].lower().startswith("rtsp://"): QMessageBox.warning(self,"Input Err","Valid RTSP URL required.");return None;
            cfg['host']=None; cfg['port']=None; cfg['user']=None; cfg['password']=None # Clear ONVIF fields
        try: cfg['motion_threshold']=max(0,int(self.motion_thresh_input.text().strip() or '0'))
        except: QMessageBox.warning(self,"Input Err","Invalid Motion Thresh.");return None;
        return cfg


# --- CameraThread Class --- (Full implementation)
class CameraThread(QThread):
    new_frame=pyqtSignal(str,object);motion_detected_signal=pyqtSignal(str);connection_status=pyqtSignal(str,bool,str)
    def __init__(self,cam:SecurityCamera,parent=None): super().__init__(parent);self.cam=cam;self._run=True;self._pause=False;self._last_st:Optional[bool]=None;self._last_err:Optional[str]=None;self.setObjectName(f"Cam_{self.cam.name}")
    def run(self):
        logger.info(f"{self.objectName()} started."); last_emit=0;interval=5;fps=15;slp=0.005;errs=0
        while self._run:
            if self._pause:time.sleep(0.5);continue; start=time.time();frame=None
            try:
                frame=self.cam.get_frame();conn=self.cam.is_connected;err=self.cam.last_error;chg=(conn!=self._last_st or err!=self._last_err);t_emit=time.time()-last_emit
                if chg or t_emit>interval or not conn:
                    try:
                        if not self.connection_status.signalsBlocked(): self.connection_status.emit(self.cam.name,conn,err or "")
                    except RuntimeError as e:logger.warning(f"Emit status err {self.cam.name}:{e}");self._run=False;break;
                    self._last_st=conn;self._last_err=err;last_emit=time.time()
                if frame is not None:
                    errs=0;
                    try:
                        if not self.new_frame.signalsBlocked(): self.new_frame.emit(self.cam.name,frame)
                    except RuntimeError as e:logger.warning(f"Emit frame err {self.cam.name}:{e}");self._run=False;break
                    if self.cam.motion_threshold>0 and self.cam.detect_motion(frame):
                        try:
                            if not self.motion_detected_signal.signalsBlocked(): self.motion_detected_signal.emit(self.cam.name)
                        except RuntimeError as e:logger.warning(f"Emit motion err {self.cam.name}:{e}");self._run=False;break
                    pt=time.time()-start;st=max(slp,(1.0/fps)-pt);time.sleep(st)
                else: # Frame is None
                    errs+=1;sd=min(1.0*errs,10.0);logger.debug(f"Frame fail({self.cam.name},{errs}x).Sleep {sd:.1f}s.");time.sleep(sd)
            except Exception as e: # Exception in main loop logic
                import traceback;terr=f"Loop err {self.objectName()}:{e}\n{traceback.format_exc()}";logger.error(terr);errs+=1
                try:
                    if self._last_err!=terr and not self.connection_status.signalsBlocked(): self.connection_status.emit(self.cam.name,False,terr);self._last_st=False;self._last_err=terr;last_emit=time.time()
                except RuntimeError:pass; except Exception as ee:logger.error(f"Emit loop err status fail:{ee}");
                time.sleep(max(5.0,1.0*errs))
        logger.info(f"{self.objectName()} stopping...");self.cam.release();logger.info(f"{self.objectName()} finished.")
    def stop(self):logger.debug(f"Stop {self.objectName()}...");self._run=False
    def pause(self):logger.debug(f"Pause {self.objectName()}");self._pause=True
    def resume(self):logger.debug(f"Resume {self.objectName()}");self._pause=False

# --- NotificationManager Class --- (Full implementation)
class NotificationManager(QLabel):
    def __init__(self,parent): super().__init__(parent);self.parent_widget=parent;self.setAlignment(Qt.AlignmentFlag.AlignCenter);self.setStyleSheet("QLabel{background-color:rgba(0,0,0,0.8);color:white;border-radius:6px;padding:12px 18px;font-size:10pt;border:1px solid #555;}");self.setWordWrap(True);self.setMinimumWidth(300);self.setMaximumWidth(500);self.setSizePolicy(QSizePolicy.Policy.Fixed,QSizePolicy.Policy.MinimumExpanding);self.hide(); self.anim=QPropertyAnimation(self,b"geometry",self);self.anim.setDuration(400);self.anim.setEasingCurve(QEasingCurve.Type.OutCubic); self._hide_tmr=QTimer(self);self._hide_tmr.setSingleShot(True);self._hide_tmr.timeout.connect(self.hide_notification);self.anim.finished.connect(self._on_animation_finished)
    def show_message(self,msg:str,dur:int=4000,lvl:str="info"):
        self.setTextFormat(Qt.TextFormat.RichText if'<'in msg and'>'in msg else Qt.TextFormat.PlainText);self.setText(msg); base="color:white;border-radius:6px;padding:12px 18px;font-size:10pt;";ls=""
        if lvl=="error":ls="background-color:rgba(231,76,60,0.9);border:1px solid #c0392b;"
        elif lvl=="warning":ls="background-color:rgba(243,156,18,0.9);border:1px solid #d35400;"
        elif lvl=="success":ls="background-color:rgba(46,204,113,0.9);border:1px solid #27ae60;"
        else:ls="background-color:rgba(52,152,219,0.9);border:1px solid #2980b9;"
        self.setStyleSheet(f"QLabel{{ {base} {ls} }}");self.adjustSize();mh=self.fontMetrics().height()*2+24;fh=max(self.height(),mh);pw=self.parent_widget.width();mw=self.width();max_w=pw-40
        if mw>max_w:mw=max_w;self.setFixedWidth(mw);self.adjustSize();fh=max(self.height(),mh); self.setFixedSize(mw,fh);sx=(pw-mw)//2;sy=-fh-10;ex=sx;ey=20;sg=QRect(sx,sy,mw,fh);eg=QRect(ex,ey,mw,fh); self.anim.stop();self._hide_tmr.stop();self.setGeometry(sg);self.show();self.raise_();self.anim.setDirection(QPropertyAnimation.Direction.Forward);self.anim.setStartValue(sg);self.anim.setEndValue(eg);self.anim.start();self._hide_tmr.start(dur)
    def hide_notification(self):
        if not self.isVisible() or (self.anim.state()==QPropertyAnimation.State.Running and self.anim.direction()==QPropertyAnimation.Direction.Backward):return; sg=self.geometry();eg=QRect(sg.x(),-sg.height()-10,sg.width(),sg.height());self._hide_tmr.stop();self.anim.stop();self.anim.setDirection(QPropertyAnimation.Direction.Backward);self.anim.setStartValue(sg);self.anim.setEndValue(eg);self.anim.start()
    @pyqtSlot()
    def _on_animation_finished(self):
        if self.anim.direction()==QPropertyAnimation.Direction.Backward:self.hide()

# --- CameraMarkerItem Class --- (Full implementation)
class CameraMarkerItem(QGraphicsPixmapItem):
    markerClicked=pyqtSignal(str); markerMoved=pyqtSignal(str,QPointF)
    def __init__(self,name:str,icon:QPixmap,pos:QPointF,parent:QGraphicsItem=None): super().__init__(icon,parent);self.camera_name=name;self.setPos(pos);self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable);self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges);self.setAcceptHoverEvents(True);self.setToolTip(f"Cam:{name}\nClick view");self.setOffset(-icon.width()/2,-icon.height());self.setCursor(Qt.CursorShape.PointingHandCursor);self._edit_mode=False;self._drag_start=QPointF();sh=QGraphicsDropShadowEffect();sh.setBlurRadius(8);sh.setColor(QColor(0,0,0,100));sh.setOffset(2,2);self.setGraphicsEffect(sh)
    def setEditMode(self,edit:bool):
        if self._edit_mode==edit:return;self._edit_mode=edit;self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable,edit);cur=Qt.CursorShape.OpenHandCursor if edit else Qt.CursorShape.PointingHandCursor;tip=f"Cam:{self.camera_name}\n{'Drag' if edit else 'Click'}";self.setCursor(cur);self.setToolTip(tip)
    def mousePressEvent(self,ev:'QGraphicsSceneMouseEvent'):
        if ev.button()==Qt.MouseButton.LeftButton:
            if self._edit_mode:self.setCursor(Qt.CursorShape.ClosedHandCursor);self._drag_start=self.pos()
            super().mousePressEvent(ev)
        else:ev.ignore()
    def mouseMoveEvent(self,ev:'QGraphicsSceneMouseEvent'):
        if self._edit_mode and ev.buttons()&Qt.MouseButton.LeftButton:super().mouseMoveEvent(ev)
        else:ev.ignore()
    def mouseReleaseEvent(self,ev:'QGraphicsSceneMouseEvent'):
        if ev.button()==Qt.MouseButton.LeftButton:
            if self._edit_mode:
                self.setCursor(Qt.CursorShape.OpenHandCursor);np=self.pos();diff=np-self._drag_start;
                if diff.manhattanLength()>QApplication.startDragDistance()/2.0:
                    logger.debug(f"Marker '{self.camera_name}' moved. Emit.");
                    self.markerMoved.emit(self.camera_name,np);
                    self.setToolTip(f"Cam:{self.camera_name}\nDrag\nPos:({np.x():.0f},{np.y():.0f})")
                else: # Treat as click if not moved significantly
                    if(ev.scenePos()-ev.buttonDownScenePos(Qt.MouseButton.LeftButton)).manhattanLength()<QApplication.startDragDistance():
                        logger.debug(f"Marker '{self.camera_name}' clicked (during edit mode).");
                        # Optionally trigger click even in edit mode, or do nothing
                        # self.markerClicked.emit(self.camera_name)
                        pass
            else: # Not in edit mode, always treat as click
                if(ev.scenePos()-ev.buttonDownScenePos(Qt.MouseButton.LeftButton)).manhattanLength()<QApplication.startDragDistance():
                    logger.debug(f"Marker '{self.camera_name}' clicked.");
                    self.markerClicked.emit(self.camera_name)

            super().mouseReleaseEvent(ev)
        else:ev.ignore()
    def hoverEnterEvent(self,ev:'QGraphicsSceneHoverEvent'): p=self.pos();b=f"Cam:{self.camera_name}\n{'Drag' if self._edit_mode else 'Click'}";self.setToolTip(f"{b}\nPos:({p.x():.0f},{p.y():.0f})");super().hoverEnterEvent(ev)
    def hoverLeaveEvent(self,ev:'QGraphicsSceneHoverEvent'): b=f"Cam:{self.camera_name}\n{'Drag' if self._edit_mode else 'Click'}";self.setToolTip(b);super().hoverLeaveEvent(ev)
    def itemChange(self,chg:QGraphicsItem.GraphicsItemChange,val:Any)->Any:
        if self._edit_mode and chg==QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            np=val;sr=self.scene().sceneRect();
            if sr.isValid() and not sr.isEmpty(): pr=self.pixmap().rect();hw=pr.width()/2.0;fh=pr.height(); cx=max(sr.left()+hw,min(np.x(),sr.right()-hw)); cy=max(sr.top()+fh,min(np.y(),sr.bottom())); return QPointF(cx,cy)
            else:return val
        return super().itemChange(chg,val)


# ==================== MAIN APPLICATION WINDOW ====================
class SecurityMonitorApp(QMainWindow):
    def __init__(self):
        super().__init__(); logger.info("Initializing App..."); self.setWindowTitle("Security Monitor Pro"); self.setGeometry(100,100,1500,950); self.setMinimumSize(1000,700); self._def_app_icon=self._create_default_icon("app"); self.setWindowIcon(self._def_app_icon)
        self.cfg_path:str="config.yaml"; self.app_cfg:Dict[str,Any]={"cameras":[],"siem":{"api_url":"","token":"","queries":[],"auth_header":"Bearer","verify_ssl":False,"refresh_interval_min":15},"soar":{"enabled":False,"api_url":"","api_key":"","auth_header_name":"Authorization","auth_header_prefix":"Bearer ","verify_ssl":True,"playbooks":[]},"map_view":{"image_path":None,"camera_positions":{}}}; self._dirty:bool=False
        self.cams:Dict[str,SecurityCamera]={}; self.cam_threads:Dict[str,CameraThread]={}; self.vid_widgets:Dict[str,QLabel]={}; self.stat_labels:Dict[str,QLabel]={}; self.mot_inds:Dict[str,QLabel]={}; self.cam_boxes:Dict[str,QGroupBox]={}; self.ptz_ctrls:Dict[str,QWidget]={}
        self.map_scene:Optional[QGraphicsScene]=None; self.map_view:Optional[QGraphicsView]=None; self.map_bg:Optional[QGraphicsPixmapItem]=None; self.map_markers:Dict[str,CameraMarkerItem]={}; self.map_edit:bool=False; self._def_cam_icon:Optional[QPixmap]=self._create_default_icon("camera")
        self.siem:Optional[SIEMClient]=None; self.siem_timer=QTimer(self); self.siem_timer.timeout.connect(self.refresh_alerts); self.soar:Optional[SOARClient]=None; self.alerts_disp:Optional[QTextEdit]=None; self._alerts_data:List[Dict]=[]
        self.load_config(self.cfg_path); self.init_ui(); self.apply_dark_theme(); self.init_system(); self.update_siem_timer_interval(); logger.info("App Init Complete.")

    def _create_default_icon(self, type:str="app")->Optional[Union[QIcon,QPixmap]]:
        try:
            if type=="camera": pix=QPixmap(24,24);pix.fill(Qt.GlobalColor.transparent);p=QPainter(pix);p.setRenderHint(QPainter.RenderHint.Antialiasing);p.setBrush(QColor(210,210,210));p.setPen(QPen(Qt.GlobalColor.black,1));p.drawRoundedRect(QRectF(2.5,5.5,19,13),3,3);p.setBrush(QColor(60,60,60));p.setPen(Qt.PenStyle.NoPen);p.drawEllipse(QRectF(7.5,8.5,9,7));p.end();return pix
            else: pix=QPixmap(32,32);pix.fill(Qt.GlobalColor.transparent);p=QPainter(pix);p.setRenderHint(QPainter.RenderHint.Antialiasing);p.setPen(QPen(QColor(180,180,180),2));p.setBrush(QColor(70,70,70));p.drawRoundedRect(QRectF(3.5,3.5,25,25),5,5);p.setPen(QPen(QColor(60,180,230),3));p.drawArc(QRectF(8,8,16,16),45*16,270*16);p.drawPoint(QPointF(16,16));p.end();return QIcon(pix)
        except Exception as e: logger.error(f"Icon err '{type}': {e}"); return None

    def update_siem_timer_interval(self):
        mins=self.app_cfg['siem'].get("refresh_interval_min",15)
        try: ms=int(mins)*60*1000;
        except: logger.warning(f"Invalid SIEM interval:'{mins}'.Using 15."); ms=15*60*1000; self.app_cfg['siem']['refresh_interval_min']=15; # Correct config if invalid
        if self.siem_timer.isActive(): self.siem_timer.stop()
        if ms>0 and self.siem and self.siem.is_configured: self.siem_timer.start(ms); logger.info(f"SIEM timer started ({mins} min).")
        elif not (self.siem and self.siem.is_configured): logger.info("SIEM timer off (SIEM N/A).")
        else: logger.info("SIEM timer off (interval<=0).")
        # Ensure UI reflects the potentially corrected value
        if hasattr(self,'siem_refresh_input'): self.siem_refresh_input.setText(str(self.app_cfg['siem']['refresh_interval_min']))


    def load_config(self, fp:str):
        logger.info(f"Loading config: {fp}"); self.cfg_path=fp; loaded=None
        try:
            if os.path.exists(fp):
                with open(fp,'r',encoding='utf-8') as f: loaded=yaml.safe_load(f)
            else:
                logger.info(f"Config file '{fp}' not found. Creating default config.")
                self._validate_and_correct_config() # Ensure self.app_cfg has defaults
                if self.save_config(fp): logger.info(f"Default config saved to '{fp}'.")
                else: logger.error(f"Failed to save initial default config to '{fp}'.")
                return # Use the defaults already in self.app_cfg

            if loaded and isinstance(loaded,dict):
                def merge(d, l):
                    m=d.copy();
                    for k,v in l.items():
                        if k in m:
                            if isinstance(m[k],dict) and isinstance(v,dict): m[k]=merge(m[k],v)
                            elif isinstance(m[k],list) and isinstance(v,list) and k in ['cameras','queries','playbooks']: m[k]=v # Replace lists entirely
                            elif v is not None and type(v) == type(m[k]): m[k]=v # Update value if type matches
                            elif v is not None and m[k] is None: m[k] = v # Allow setting if original was None
                            elif v is not None: logger.warning(f"Cfg type mismatch for key '{k}'. Using loaded value type {type(v)} over default {type(m[k])}."); m[k]=v
                        else: m[k]=v # Add new key
                    return m
                # Start with default structure, then merge loaded config onto it
                default_cfg = {"cameras":[],"siem":{"api_url":"","token":"","queries":[],"auth_header":"Bearer","verify_ssl":False,"refresh_interval_min":15},"soar":{"enabled":False,"api_url":"","api_key":"","auth_header_name":"Authorization","auth_header_prefix":"Bearer ","verify_ssl":True,"playbooks":[]},"map_view":{"image_path":None,"camera_positions":{}}}
                self.app_cfg=merge(default_cfg, loaded)
                logger.info(f"Config loaded & merged from: {fp}")
            elif loaded is None: logger.warning(f"Config file '{fp}' was empty. Using defaults.")
            else: logger.error(f"Config structure in '{fp}' is invalid (not a dictionary). Using defaults.")
            # Always validate after loading or using defaults
            self._validate_and_correct_config()
        except yaml.YAMLError as e: logger.error(f"Config parse err: {fp}: {e}",exc_info=True); self._cfg_load_err_popup(fp,e); self._validate_and_correct_config() # Fallback to defaults on parse error
        except Exception as e: logger.error(f"Config load err: {fp}: {e}",exc_info=True); self._cfg_load_err_popup(fp,e); self._validate_and_correct_config() # Fallback to defaults on other errors


    def _cfg_load_err_popup(self, fp, err):
         if hasattr(self,'central_widget') and self.central_widget: QMessageBox.critical(self,"Config Error",f"Error loading/parsing config:\n{fp}\n\n{err}\n\nFalling back to defaults or previous state.")

    def _validate_and_correct_config(self):
        logger.debug("Validating and correcting config...")
        # Ensure top-level keys exist
        for k in ['cameras','siem','soar','map_view']: self.app_cfg.setdefault(k,{})
        # --- Cameras ---
        if not isinstance(self.app_cfg.get('cameras'),list): self.app_cfg['cameras']=[]
        valid_cams=[]; used_names=set();
        for i,c in enumerate(self.app_cfg['cameras']):
            if not isinstance(c,dict): continue;
            original_name = str(c.get('name','')).strip(); valid_name = original_name if original_name else f"UnnamedCamera_{i+1}"; unique_name=valid_name; count=1
            while unique_name in used_names: unique_name=f"{valid_name}_{count}"; count+=1
            if unique_name != original_name: logger.warning(f"Camera name adjusted: '{original_name}' -> '{unique_name}'")
            used_names.add(unique_name); c['name'] = unique_name
            c['onvif'] = bool(c.get('onvif',False));
            c['url'] = str(c.get('url')) or None;
            c['host'] = str(c.get('host')) or None;
            try: c['port']=int(c.get('port')) if c.get('port') is not None else None; assert c['port'] is None or (1<=c['port']<=65535)
            except: c['port']=80 if c['onvif'] else None; logger.warning(f"Corrected invalid port for '{unique_name}' to {c['port']}.")
            c['user']=str(c.get('user'))or None;c['password']=str(c.get('password'))or None;
            try:c['motion_threshold']=max(0,int(c.get('motion_threshold',500)));
            except:c['motion_threshold']=500; logger.warning(f"Corrected invalid motion threshold for '{unique_name}' to {c['motion_threshold']}.")
            # Basic logic check
            if c['onvif'] and not (c['host'] and c['port']): logger.warning(f"ONVIF camera '{unique_name}' missing host/port. Connection may fail.");
            elif not c['onvif'] and not c['url']: logger.warning(f"RTSP camera '{unique_name}' missing URL. Connection will fail.")
            valid_cams.append(c);
        self.app_cfg['cameras']=valid_cams; current_cam_names={c['name'] for c in valid_cams}

        # --- SIEM ---
        if not isinstance(self.app_cfg.get('siem'),dict): self.app_cfg['siem']={'queries':[]}
        sc=self.app_cfg['siem']; sc.setdefault('queries',[]); sc.setdefault('auth_header','Bearer'); sc.setdefault('verify_ssl',False); sc.setdefault('refresh_interval_min',15);
        sc['api_url']=str(sc.get('api_url','')); sc['token']=str(sc.get('token',''));
        sc['auth_header']=str(sc.get('auth_header','Bearer')); sc['verify_ssl']=bool(sc.get('verify_ssl',False));
        try: sc['refresh_interval_min']=max(0,int(sc.get('refresh_interval_min')));
        except: sc['refresh_interval_min']=15; logger.warning("Corrected invalid SIEM refresh interval to 15.")
        valid_queries=[];
        if not isinstance(sc.get('queries'),list): sc['queries']=[]
        for q in sc['queries']:
            if isinstance(q,dict) and q.get('name') and q.get('query'):
                 q['name']=str(q['name']); q['query']=str(q['query']); q['enabled']=bool(q.get('enabled',True)); valid_queries.append(q)
            else: logger.warning(f"Skipping invalid SIEM query: {q}")
        sc['queries']=valid_queries

        # --- SOAR ---
        if not isinstance(self.app_cfg.get('soar'),dict): self.app_cfg['soar']={'playbooks':[]}
        soc=self.app_cfg['soar']; soc.setdefault('enabled',False); soc.setdefault('auth_header_name','Authorization'); soc.setdefault('auth_header_prefix','Bearer '); soc.setdefault('verify_ssl',True); soc.setdefault('playbooks',[]);
        soc['enabled']=bool(soc.get('enabled')); soc['api_url']=str(soc.get('api_url','')); soc['api_key']=str(soc.get('api_key',''));
        soc['auth_header_name']=str(soc.get('auth_header_name','Authorization')); soc['auth_header_prefix']=str(soc.get('auth_header_prefix','Bearer ')); soc['verify_ssl']=bool(soc.get('verify_ssl',True));
        valid_playbooks=[];
        if not isinstance(soc.get('playbooks'),list): soc['playbooks']=[]
        for p in soc['playbooks']:
            if isinstance(p,dict) and p.get('name') and p.get('id'):
                 p['name']=str(p['name']); p['id']=str(p['id']); ctx=p.get('context_fields',[]); p['context_fields']=[str(f) for f in ctx if isinstance(f,str)] if isinstance(ctx,list) else []; valid_playbooks.append(p)
            else: logger.warning(f"Skipping invalid SOAR playbook: {p}")
        soc['playbooks']=valid_playbooks

        # --- Map View ---
        if not isinstance(self.app_cfg.get('map_view'),dict): self.app_cfg['map_view']={'camera_positions':{}}
        mc=self.app_cfg['map_view']; mc.setdefault('camera_positions',{});
        mc['image_path']=str(mc.get('image_path')) or None;
        if mc['image_path'] and not os.path.exists(mc['image_path']):
            logger.warning(f"Map image path invalid, clearing: {mc['image_path']}")
            mc['image_path'] = None
        if not isinstance(mc.get('camera_positions'),dict):mc['camera_positions']={}
        valid_positions = {}
        for name, pos_data in mc['camera_positions'].items():
            if name not in current_cam_names: logger.info(f"Pruning stale map position for deleted camera: {name}"); continue
            if not isinstance(pos_data,dict) or 'x' not in pos_data or 'y' not in pos_data: logger.warning(f"Invalid map position data structure for '{name}', removing."); continue
            try: valid_positions[name]={'x':float(pos_data['x']), 'y':float(pos_data['y'])}
            except (ValueError, TypeError): logger.warning(f"Non-numeric map position coordinates for '{name}', removing."); continue
        mc['camera_positions'] = valid_positions
        logger.debug("Config validation and correction complete.")


    def save_config(self, fp: Optional[str]=None) -> bool:
        sp=fp or self.cfg_path; logger.info(f"Saving config: {sp}")
        try:
            # Update camera positions from map markers if map exists
            if self.map_markers and 'map_view' in self.app_cfg:
                 current_positions={}; active_cam_names={c.get('name') for c in self.app_cfg.get('cameras',[])}
                 for name, marker in self.map_markers.items():
                     if name in active_cam_names: current_positions[name]={'x':round(marker.pos().x(),2),'y':round(marker.pos().y(),2)}
                     else: logger.warning(f"During save: Skipping map marker for unknown/deleted camera '{name}'.")
                 self.app_cfg['map_view']['camera_positions']=current_positions

            # Ensure target directory exists
            save_dir=os.path.dirname(sp);
            if save_dir and not os.path.exists(save_dir): os.makedirs(save_dir,exist_ok=True); logger.info(f"Created directory for config: {save_dir}")

            # Write the config file
            with open(sp,'w',encoding='utf-8') as f: yaml.safe_dump(self.app_cfg,f,default_flow_style=False,sort_keys=False,allow_unicode=True,indent=2)
            logger.info(f"Config saved successfully: {sp}"); self.mark_settings_dirty(False); return True
        except yaml.YAMLError as e: logger.error(f"YAML save error: {e}",exc_info=True); if hasattr(self,'notifications'):self.notifications.show_message(f"YAML Save Error: {e}",level="error"); return False
        except Exception as e: logger.error(f"Save config error: {e}",exc_info=True); if hasattr(self,'notifications'):self.notifications.show_message(f"Save Config Error: {e}",level="error"); return False

    def mark_settings_dirty(self, dirty=True):
        if self._dirty != dirty: self._dirty=dirty; self.update_window_title(); logger.debug(f"Settings dirty flag set to: {dirty}")

    def update_window_title(self):
        base_title="SecMon Pro"; config_filename=os.path.basename(self.cfg_path) if self.cfg_path else "Untitled"; title=f"{base_title} - {config_filename}{' *' if self._dirty else ''}"; self.setWindowTitle(title)

    def init_ui(self):
        logger.debug("Initializing UI..."); self.update_window_title(); self.central_widget=QWidget(); self.setCentralWidget(self.central_widget); self.main_layout=QVBoxLayout(self.central_widget); self.main_layout.setContentsMargins(5,5,5,5); self.main_layout.setSpacing(5); self.init_menu_bar(); self.status_bar=QStatusBar(); self.setStatusBar(self.status_bar); self.status_bar.showMessage("Initializing UI..."); self.notifications=NotificationManager(self.central_widget); self.tabs=QTabWidget(); self.tabs.currentChanged.connect(self.on_tab_changed); self.main_layout.addWidget(self.tabs); self.create_monitor_tab(); self.create_map_view_tab(); self.create_settings_tab(); self.tabs.setCurrentIndex(0); self.status_bar.showMessage("UI Initialization Complete.", 3000); logger.debug("UI Init Complete.")

    def init_menu_bar(self):
        mb=self.menuBar(); fm=mb.addMenu("&File"); ld=QAction(QIcon.fromTheme("document-open"),"&Load Config...",self); ld.setShortcut("Ctrl+O"); ld.setStatusTip("Load configuration from a YAML file"); ld.triggered.connect(self.load_config_dialog); fm.addAction(ld); sv=QAction(QIcon.fromTheme("document-save"),"&Save Config",self); sv.setShortcut("Ctrl+S"); sv.setStatusTip("Save current configuration"); sv.triggered.connect(lambda:self.save_config()); fm.addAction(sv); sa=QAction(QIcon.fromTheme("document-save-as"),"Save Config &As...",self); sa.setStatusTip("Save current configuration to a new file"); sa.triggered.connect(self.save_config_dialog); fm.addAction(sa); fm.addSeparator(); ex=QAction(QIcon.fromTheme("application-exit"),"E&xit",self); ex.setShortcut("Ctrl+Q"); ex.setStatusTip("Exit the application"); ex.triggered.connect(self.close); fm.addAction(ex); hm=mb.addMenu("&Help"); ab=QAction(QIcon.fromTheme("help-about"),"&About",self); ab.setStatusTip("Show application information"); ab.triggered.connect(self.show_about); hm.addAction(ab)

    def on_tab_changed(self, index:int): logger.debug(f"Switched to tab: {self.tabs.tabText(index)} (Index: {index})")

    def load_config_dialog(self):
        if self.check_unsaved_changes("load a new configuration"):
            current_dir=os.path.dirname(self.cfg_path) if self.cfg_path else "";
            fp, _ = QFileDialog.getOpenFileName(self,"Load Configuration File",current_dir,"YAML Files (*.yaml *.yml);;All Files (*)")
            if fp:
                self.stop_all_cameras(); # Stop before loading new config
                self.load_config(fp);    # Load the selected file
                self.refresh_settings_ui(); # Update settings tab UI
                self.init_system();         # Re-initialize cameras, SIEM, SOAR with new config
                self.update_siem_timer_interval(); # Reset SIEM timer
                self.recreate_monitor_tab(); # Rebuild monitor tab UI
                self.load_map_image();       # Reload map image based on new config
                self.mark_settings_dirty(False); # Config is now saved (or freshly loaded)
                self.notifications.show_message(f"Configuration loaded: {os.path.basename(fp)}.",level="success")

    def save_config_dialog(self):
        suggested_path=self.cfg_path or "config.yaml";
        fp, _ = QFileDialog.getSaveFileName(self,"Save Configuration As",suggested_path,"YAML Files (*.yaml *.yml);;All Files (*)")
        if fp:
            if self.save_config(fp):
                self.cfg_path=fp; # Update the current config path
                self.update_window_title(); # Remove dirty marker '*'
                self.notifications.show_message(f"Configuration saved to: {os.path.basename(fp)}",level="success")
            else:
                self.notifications.show_message(f"Failed to save configuration to: {os.path.basename(fp)}",level="error") # save_config logs details


    def stop_all_cameras(self):
        thread_count = len(self.cam_threads)
        if thread_count == 0:
            logger.info("No active camera threads to stop.")
        else:
            logger.info(f"Stopping {thread_count} camera thread(s)...")
            threads_to_stop = list(self.cam_threads.values())
            for t in threads_to_stop:
                if t and t.isRunning():
                    t.stop() # Signal thread to stop

            start_time = time.time()
            max_wait_seconds = 8.0
            still_running = threads_to_stop[:] # Copy list

            while still_running and (time.time() - start_time) < max_wait_seconds:
                still_running = [t for t in still_running if t and t.isRunning()]
                if still_running:
                    time.sleep(0.05)
                    QApplication.processEvents() # Keep UI responsive

            if still_running:
                logger.warning(f"{len(still_running)} camera thread(s) did not stop gracefully within {max_wait_seconds} seconds.")
                # Optionally, could force termination here if needed, but usually wait() is better
                # for t in still_running: t.terminate() # Use with caution
            else:
                logger.info("All camera threads stopped successfully.")

        # Clear all camera-related state dictionaries and widgets
        self.cams.clear(); self.cam_threads.clear(); self.vid_widgets.clear();
        self.stat_labels.clear(); self.mot_inds.clear(); self.cam_boxes.clear(); self.ptz_ctrls.clear();

        # Clear markers from the map scene
        if self.map_scene:
            markers_to_remove = list(self.map_markers.values())
            for marker in markers_to_remove:
                if marker and marker.scene() == self.map_scene:
                    self.map_scene.removeItem(marker)
            self.map_markers.clear()
            logger.info("Cleared map markers.")
        else:
             self.map_markers.clear() # Clear dict even if scene doesn't exist

        logger.info("Cleared internal camera state.")


    def init_system(self):
        logger.info("Initializing system components (Cameras, SIEM, SOAR)...");
        QApplication.processEvents(); # Ensure UI is responsive during init
        self.stop_all_cameras(); # Ensure previous state is cleared

        # Stop SIEM Timer if running
        if self.siem_timer.isActive(): self.siem_timer.stop()
        self.siem=None; self.soar=None; # Reset clients
        logger.info("Stopped existing components.")

        # Initialize Cameras
        camera_errors=[]; used_names=set(); camera_configs = self.app_cfg.get('cameras',[])
        logger.info(f"Initializing {len(camera_configs)} camera(s) from config...")
        for i, cfg in enumerate(camera_configs):
            cam_name = cfg.get('name');
            if not cam_name: logger.error(f"Skipping camera config at index {i} due to missing name."); continue;
            if cam_name in self.cams: logger.warning(f"Camera name '{cam_name}' already exists, skipping duplicate initialization."); continue;
            used_names.add(cam_name)
            try:
                cam = SecurityCamera(cfg); self.cams[cam_name]=cam;
                thread = CameraThread(cam, self);
                thread.new_frame.connect(self.update_video_frame,Qt.ConnectionType.QueuedConnection);
                thread.motion_detected_signal.connect(self.on_motion_detected,Qt.ConnectionType.QueuedConnection);
                thread.connection_status.connect(self.on_camera_connection_status,Qt.ConnectionType.QueuedConnection);
                self.cam_threads[cam_name]=thread;
                thread.start()
                logger.debug(f"Started thread for camera '{cam_name}'.")
            except Exception as e:
                import traceback; err_msg=f"Camera initialization error for '{cam_name}': {e}\n{traceback.format_exc()}"; logger.error(err_msg); camera_errors.append(f"{cam_name}: Init Error");
                # Clean up partially initialized state if error occurred
                if cam_name in self.cams: del self.cams[cam_name];
                if cam_name in self.cam_threads: thread=self.cam_threads.pop(cam_name); thread.stop(); thread.wait(500); # Wait briefly for thread exit
        logger.info(f"Camera initialization finished. {len(self.cams)} started, {len(camera_errors)} errors.")
        if camera_errors: logger.warning(f"Camera errors: {'; '.join(camera_errors)}")

        # Initialize SIEM Client
        try:
            self.siem=SIEMClient(self.app_cfg.get('siem',{}))
            if self.siem.is_configured:
                logger.info("SIEM client initialized successfully and is configured.");
                QTimer.singleShot(1000,self.refresh_alerts) # Initial fetch after 1 sec
            else:
                 logger.warning("SIEM client initialized but is not configured (missing URL/token or no enabled queries).");
                 if self.alerts_disp: self.alerts_disp.setPlainText("SIEM not configured or no queries enabled.")
        except Exception as e: logger.error(f"SIEM client initialization failed: {e}",exc_info=True); if hasattr(self,'notifications'): self.notifications.show_message(f"SIEM Initialization Error: {e}",level="error")

        # Initialize SOAR Client
        try:
            self.soar=SOARClient(self.app_cfg.get('soar',{}))
            if self.soar.is_configured and self.app_cfg.get('soar',{}).get('enabled',False):
                logger.info("SOAR client initialized successfully and is enabled.")
            elif self.soar.is_configured:
                logger.info("SOAR client initialized but is disabled in configuration.")
            else:
                 logger.warning("SOAR client initialized but is not configured (missing URL/API key).")
        except Exception as e: logger.error(f"SOAR client initialization failed: {e}",exc_info=True); if hasattr(self,'notifications'): self.notifications.show_message(f"SOAR Initialization Error: {e}",level="error")

        # Update map markers based on initialized cameras
        self.update_map_markers()

        # Final status message
        num_cams=len(self.cams); status_msg="System ready.";
        if camera_errors: status_msg=f"{num_cams} camera(s) started. Errors: {len(camera_errors)}."
        elif num_cams==0: status_msg="System ready. No cameras configured or started."
        else: status_msg=f"System ready. {num_cams} camera(s) running."
        if self.soar and self.soar.is_configured and self.app_cfg.get('soar',{}).get('enabled'): status_msg+=" SOAR enabled." else: status_msg+=" SOAR disabled."
        if hasattr(self,'status_bar'): self.status_bar.showMessage(status_msg,5000);
        logger.info(f"System initialization finished: {status_msg}")

    @pyqtSlot()
    def refresh_alerts(self):
        if not self.alerts_disp: logger.warning("Alert display not available, cannot refresh."); return
        if not self.siem or not self.siem.is_configured:
            self.alerts_disp.setPlainText("SIEM not configured or no queries enabled.")
            logger.info("SIEM refresh skipped: Not configured.")
            if hasattr(self,'refresh_btn'): self.refresh_btn.setEnabled(True) # Ensure enabled if clicked manually
            return

        logger.debug("Starting SIEM alert refresh...");
        # Visual feedback: change border or disable button
        original_style = self.alerts_disp.styleSheet()
        self.alerts_disp.setStyleSheet(original_style + " border-color: yellow;")
        if hasattr(self,'refresh_btn'): self.refresh_btn.setEnabled(False);
        QApplication.processEvents() # Update UI immediately

        def fetch_task():
            alerts = self.siem.fetch_alerts() # This blocks in the thread
            # Schedule UI updates back on the main thread
            QMetaObject.invokeMethod(self,"_update_alerts_display",Qt.ConnectionType.QueuedConnection,Q_ARG(list,alerts))
            QMetaObject.invokeMethod(self,"_reenable_refresh_button",Qt.ConnectionType.QueuedConnection)
            # Restore original style after completion (also on main thread)
            QMetaObject.invokeMethod(self.alerts_disp, "setStyleSheet", Qt.ConnectionType.QueuedConnection, Q_ARG(str, original_style))

        # Run the fetch task in a separate thread to avoid blocking the GUI
        threading.Thread(target=fetch_task,daemon=True).start()

    @pyqtSlot()
    def _reenable_refresh_button(self):
        if hasattr(self,'refresh_btn'): self.refresh_btn.setEnabled(True)
        # Note: Style restoration moved to the fetch_task completion for reliability

    @pyqtSlot(list)
    def _update_alerts_display(self, alerts:List[Dict]):
        if not self.alerts_disp: return
        logger.debug(f"Updating alert display with {len(alerts)} events.");
        self._alerts_data=alerts; # Store data for context menu
        self.alerts_disp.clear()

        if not alerts: self.alerts_disp.setPlainText("No SIEM events found matching criteria."); return

        # Build HTML display
        html_parts=["<style>p{margin-bottom:4px;line-height:1.3;border-bottom:1px dotted #444;padding-bottom:4px;} b{color:#aaddff;} .qn{color:#FFD700;font-weight:bold;display:block;margin-bottom:3px;} .raw{font-family:Consolas,'Courier New',monospace;white-space:pre-wrap;color:#ccc;font-size:8pt;display:block;background-color:#2f2f2f;padding:4px;border-radius:3px;margin-top:3px;}</style>"]
        max_display=150; displayed_alerts=alerts[:max_display]; skipped_count=len(alerts)-len(displayed_alerts)

        for idx,alert in enumerate(displayed_alerts):
            query_name=alert.get('_query_name','Unknown Query'); html_parts.append(f"<p data-idx='{idx}'>"); html_parts.append(f"<span class='qn'>Source Query: {query_name}</span>")
            timestamp_str=alert.get('_time',''); display_time=""
            if timestamp_str:
                try: # Attempt parsing various common formats
                    ts_cleaned=str(timestamp_str).strip().split('+')[0].split('Z')[0].split('.')[0] # Clean common suffixes
                    parsed = False
                    formats_to_try = ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%b %d %H:%M:%S", "%m/%d/%Y %H:%M:%S"]
                    for fmt in formats_to_try:
                        try: dt_obj=datetime.datetime.strptime(ts_cleaned,fmt); display_time=dt_obj.strftime('%Y-%m-%d %H:%M:%S'); parsed=True; break;
                        except ValueError: pass
                    if not parsed: # Try epoch time
                        try: dt_obj=datetime.datetime.fromtimestamp(float(timestamp_str)); display_time=dt_obj.strftime('%Y-%m-%d %H:%M:%S'); parsed=True;
                        except ValueError: pass
                    if not parsed: display_time=str(timestamp_str) # Fallback to original string
                except Exception as e_ts: display_time=str(timestamp_str); logger.warning(f"Timestamp parse error for '{timestamp_str}': {e_ts}")
                html_parts.append(f"<b>Time:</b> {display_time}<br>")

            # Display key fields (customize this list as needed)
            key_fields=['host','src_ip','dest_ip','user','process_name','signature','severity','action','rule_id','alert_name','command_line','object','url']
            for field in key_fields:
                value = alert.get(field)
                if value:
                     display_value=str(value).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'); # Basic HTML escaping
                     html_parts.append(f"<b>{field.replace('_',' ').title()}:</b> {display_value}<br>")

            # Display raw or summary details
            raw_event=alert.get('_raw'); max_raw_len=400
            if raw_event:
                event_text=str(raw_event).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;');
                display_raw=event_text[:max_raw_len]+('...'if len(event_text)>max_raw_len else '');
                html_parts.append(f"<span class='raw'>{display_raw}</span>")
            else: # Fallback to JSON dump of non-key fields
                fallback_details={k:v for k,v in alert.items() if k not in ['_query_name','_time','_raw'] and k not in key_fields};
                try:
                    json_dump=json.dumps(fallback_details,indent=2).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;');
                    display_json=json_dump[:max_raw_len]+('...'if len(json_dump)>max_raw_len else '');
                    html_parts.append(f"<span class='raw'>{display_json}</span>")
                except Exception: html_parts.append("<span class='raw'><i>(Could not display details)</i></span>")
            html_parts.append("</p>")

        if skipped_count>0: html_parts.append(f"<p><i>(Displaying first {max_display} of {len(alerts)} events)</i></p>")

        self.alerts_disp.setHtml("".join(html_parts));
        self.alerts_disp.moveCursor(QTextCursor.MoveOperation.Start) # Scroll to top
        if hasattr(self,'status_bar'): self.status_bar.showMessage(f"SIEM refreshed: {len(alerts)} event(s) found.", 4000)

    def recreate_monitor_tab(self):
        logger.info("Recreating 'Monitoring' tab...");
        monitor_tab_index=-1; current_tab_index = self.tabs.currentIndex()
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i)=="Monitoring": monitor_tab_index=i; break

        if monitor_tab_index!=-1:
            widget_to_remove=self.tabs.widget(monitor_tab_index);
            if widget_to_remove:
                self.tabs.removeTab(monitor_tab_index);
                widget_to_remove.deleteLater(); # Schedule for deletion
                logger.debug("Removed old 'Monitoring' tab and its widget.")
            else: logger.warning("Found 'Monitoring' tab index but widget was null.")

            # Clear associated UI elements and data caches before recreating
            self.vid_widgets.clear(); self.stat_labels.clear(); self.mot_inds.clear();
            self.cam_boxes.clear(); self.ptz_ctrls.clear(); self.alerts_disp=None; # Crucial: reset alerts_disp
            logger.debug("Cleared UI element caches for monitor tab.")

            self.create_monitor_tab(); # Create the new tab content and add it
            logger.debug("Created new 'Monitoring' tab content.")

            # Restore selection intelligently
            new_monitor_tab_index = 0 # It's always inserted at the beginning now
            if current_tab_index==monitor_tab_index: self.tabs.setCurrentIndex(new_monitor_tab_index) # Reselect if it was selected
            elif 0<=current_tab_index<monitor_tab_index: self.tabs.setCurrentIndex(current_tab_index) # No change needed
            elif current_tab_index>monitor_tab_index: self.tabs.setCurrentIndex(current_tab_index-1) # Adjust index
            else: self.tabs.setCurrentIndex(new_monitor_tab_index) # Default to selecting it
        else:
            logger.warning("'Monitoring' tab not found. Creating it fresh.")
            self.create_monitor_tab(); # Create if it somehow didn't exist
            self.tabs.setCurrentIndex(0) # Select the newly created tab

    def create_monitor_tab(self):
        monitor_tab_widget=QWidget();
        main_hbox_layout=QHBoxLayout(monitor_tab_widget);
        main_hbox_layout.setContentsMargins(5,5,5,5); main_hbox_layout.setSpacing(10)

        # Left side: Camera Scroll Area
        camera_scroll_area=QScrollArea();
        camera_scroll_area.setWidgetResizable(True);
        camera_scroll_area.setStyleSheet("QScrollArea{border:none;background-color:transparent;}")
        camera_container_widget=QWidget();
        camera_vbox_layout=QVBoxLayout(camera_container_widget);
        camera_vbox_layout.setContentsMargins(0,0,0,0); camera_vbox_layout.setSpacing(10)

        # Clear old widgets before creating new ones
        self.cam_boxes.clear();self.vid_widgets.clear();self.stat_labels.clear();self.mot_inds.clear();self.ptz_ctrls.clear()

        cameras_config=self.app_cfg.get('cameras',[])
        if not cameras_config:
            no_cam_label = QLabel("No cameras configured. Add cameras in the Settings tab.");
            no_cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            camera_vbox_layout.addWidget(no_cam_label)
        else:
            logger.debug(f"Creating UI widgets for {len(cameras_config)} camera(s)...");
            for cfg in cameras_config:
                cam_box = self._create_camera_widget(cfg)
                if cam_box: camera_vbox_layout.addWidget(cam_box)
            camera_vbox_layout.addStretch() # Push cameras upwards

        camera_scroll_area.setWidget(camera_container_widget)

        # Right side: SIEM Alerts Panel
        siem_groupbox=QGroupBox("SIEM Events & Actions");
        siem_vbox_layout=QVBoxLayout(siem_groupbox);
        siem_vbox_layout.setContentsMargins(8,8,8,8)

        # Create alerts display (ensure it's new if recreating)
        self.alerts_disp=QTextEdit(); # Always create a new instance here
        self.alerts_disp.setReadOnly(True);
        self.alerts_disp.setStyleSheet("background-color:#262626;color:#ddd;border:1px solid #555;font-family:Consolas,'Courier New',monospace;font-size:9pt;");
        self.alerts_disp.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap); # Keep lines from wrapping
        self.alerts_disp.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu);
        self.alerts_disp.customContextMenuRequested.connect(self.show_alert_context_menu)

        # Refresh button
        self.refresh_btn=QPushButton(QIcon.fromTheme("view-refresh")," Refresh SIEM");
        self.refresh_btn.setToolTip("Fetch latest SIEM events based on configured queries");
        self.refresh_btn.clicked.connect(self.refresh_alerts);
        self.refresh_btn.setSizePolicy(QSizePolicy.Policy.Fixed,QSizePolicy.Policy.Fixed)

        siem_vbox_layout.addWidget(self.alerts_disp, stretch=1); # Make text edit expand
        siem_vbox_layout.addWidget(self.refresh_btn, alignment=Qt.AlignmentFlag.AlignRight)

        # Add left and right panels to the main layout
        main_hbox_layout.addWidget(camera_scroll_area, stretch=65); # 65% width for cameras
        main_hbox_layout.addWidget(siem_groupbox, stretch=35);     # 35% width for SIEM

        # Insert the tab (always at position 0)
        self.tabs.insertTab(0,monitor_tab_widget,QIcon.fromTheme("utilities-log"),"Monitoring");
        logger.debug("Finished creating/updating 'Monitoring' tab.")


    def _create_camera_widget(self, cfg:dict)->Optional[QGroupBox]:
        cam_name=cfg.get('name');
        if not cam_name: logger.error("Cannot create camera widget: Config missing 'name'."); return None;

        cam_box=QGroupBox(cam_name); self.cam_boxes[cam_name] = cam_box # Store the box itself
        box_layout=QVBoxLayout(cam_box); box_layout.setContentsMargins(5,8,5,5); box_layout.setSpacing(4);

        # Top row: Video + Status/Motion
        top_hbox_layout=QHBoxLayout(); top_hbox_layout.setSpacing(8)

        # Video display label
        video_label=QLabel("Initializing...");
        video_label.setAlignment(Qt.AlignmentFlag.AlignCenter);
        video_label.setStyleSheet("background-color:#1e1e1e;color:#888;border:1px solid #444;border-radius:3px;");
        video_label.setMinimumSize(320,180); # Minimum reasonable video size
        video_label.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding);
        self.vid_widgets[cam_name]=video_label; # Store reference
        top_hbox_layout.addWidget(video_label, stretch=1) # Make video expand

        # Status indicators (vertical layout)
        status_vbox_layout=QVBoxLayout(); status_vbox_layout.setSpacing(5); status_vbox_layout.setAlignment(Qt.AlignmentFlag.AlignTop);

        # Connection status label
        status_label=QLabel("⚪ Waiting"); status_label.setToolTip("Connection Status");
        status_label.setStyleSheet("font-size:8pt;color:#aaa;");
        self.stat_labels[cam_name]=status_label; # Store reference
        status_vbox_layout.addWidget(status_label);

        # Motion indicator label
        motion_label=QLabel(); motion_label.setToolTip("Motion Detected Indicator");
        motion_label.setFixedSize(16,16);
        motion_label.setStyleSheet("background-color:transparent;border:1px solid #666;border-radius:8px;"); # Default: inactive
        self.mot_inds[cam_name]=motion_label; # Store reference
        status_vbox_layout.addWidget(motion_label, alignment=Qt.AlignmentFlag.AlignLeft);

        top_hbox_layout.addLayout(status_vbox_layout);
        box_layout.addLayout(top_hbox_layout, stretch=1) # Top row expands vertically

        # Bottom row: Controls (Snapshot, PTZ)
        controls_hbox_layout=QHBoxLayout(); controls_hbox_layout.setSpacing(5); controls_hbox_layout.setContentsMargins(0,5,0,0);

        # Snapshot button
        snapshot_button=QPushButton(QIcon.fromTheme("camera-photo"),"");
        snapshot_button.setToolTip(f"Take Snapshot ({cam_name})");
        snapshot_button.setFixedSize(30,30); snapshot_button.setIconSize(QSize(18,18));
        snapshot_button.clicked.connect(lambda checked=False, name=cam_name: self.take_snapshot(name)); # Use lambda to capture name
        controls_hbox_layout.addWidget(snapshot_button);
        controls_hbox_layout.addSpacing(15) # Space between snapshot and PTZ

        # PTZ Controls (placeholder widget)
        ptz_widget=QWidget(); ptz_layout=QHBoxLayout(ptz_widget);
        ptz_layout.setContentsMargins(0,0,0,0); ptz_layout.setSpacing(2);
        ptz_widget._ptz_buttons=[] # Custom attribute to store buttons for easy enable/disable

        if cfg.get('onvif',False): # Only create PTZ buttons if ONVIF is enabled
            ptz_buttons_dict = self._create_ptz_controls(cam_name);
            ptz_widget._ptz_buttons = list(ptz_buttons_dict.values()) # Store list of buttons
            # Add buttons to layout
            ptz_layout.addWidget(ptz_buttons_dict["left"]); ptz_layout.addWidget(ptz_buttons_dict["up"]);
            ptz_layout.addWidget(ptz_buttons_dict["down"]); ptz_layout.addWidget(ptz_buttons_dict["right"]);
            ptz_layout.addSpacing(10); # Space between pan/tilt and zoom
            ptz_layout.addWidget(ptz_buttons_dict["zoomin"]); ptz_layout.addWidget(ptz_buttons_dict["zoomout"]);
            ptz_widget.setVisible(True) # Make visible by default if ONVIF
        else:
            ptz_widget.setVisible(False) # Hide if not ONVIF

        self.ptz_ctrls[cam_name]=ptz_widget; # Store reference to the widget containing PTZ buttons
        controls_hbox_layout.addWidget(ptz_widget);
        controls_hbox_layout.addStretch(); # Push controls to the left
        box_layout.addLayout(controls_hbox_layout);

        return cam_box


    def _create_ptz_controls(self, cam_name:str)->Dict[str,QPushButton]:
        buttons:Dict[str,QPushButton]={}; icon_size=QSize(16,16); button_size=QSize(28,28); pan_tilt_speed=0.6; zoom_speed=0.5

        def create_button(key:str, icon_theme:str, tooltip:str, press_action, release_action):
            button=QPushButton(QIcon.fromTheme(icon_theme),"");
            button.setToolTip(f"{tooltip} ({cam_name})");
            button.setFixedSize(button_size); button.setIconSize(icon_size);
            button.setAutoRepeat(False); # Important for press/release logic
            button.pressed.connect(press_action);
            button.released.connect(release_action);
            button.setEnabled(False); # Initially disabled until connection confirmed
            buttons[key]=button;
            return button

        # Create buttons using the helper function
        create_button("up",    "go-up",      "Tilt Up",   lambda:self.start_ptz(cam_name, 0, pan_tilt_speed, 0), lambda:self.stop_ptz(cam_name));
        create_button("down",  "go-down",    "Tilt Down", lambda:self.start_ptz(cam_name, 0, -pan_tilt_speed, 0), lambda:self.stop_ptz(cam_name));
        create_button("left",  "go-previous","Pan Left",  lambda:self.start_ptz(cam_name, -pan_tilt_speed, 0, 0), lambda:self.stop_ptz(cam_name));
        create_button("right", "go-next",    "Pan Right", lambda:self.start_ptz(cam_name, pan_tilt_speed, 0, 0), lambda:self.stop_ptz(cam_name));
        create_button("zoomin","zoom-in",    "Zoom In",   lambda:self.start_ptz(cam_name, 0, 0, zoom_speed),   lambda:self.stop_ptz(cam_name));
        create_button("zoomout","zoom-out",  "Zoom Out",  lambda:self.start_ptz(cam_name, 0, 0, -zoom_speed),  lambda:self.stop_ptz(cam_name));

        return buttons

    @pyqtSlot(QPoint)
    def show_alert_context_menu(self, pos:QPoint):
        if not self.alerts_disp or not self._alerts_data: return # No data or display

        # Basic Copy action
        menu=QMenu(self);
        copy_action = menu.addAction(QIcon.fromTheme("edit-copy"), "Copy Selected Text");
        copy_action.triggered.connect(self.alerts_disp.copy)
        copy_action.setEnabled(self.alerts_disp.textCursor().hasSelection()) # Enable only if text is selected

        # SOAR Actions - only if SOAR is configured, enabled, and playbooks exist
        soar_configured = self.soar and self.soar.is_configured
        soar_enabled = self.app_cfg.get('soar',{}).get('enabled',False)
        playbooks = self.app_cfg.get('soar',{}).get('playbooks',[])

        if not (soar_configured and soar_enabled and playbooks):
            menu.exec(self.alerts_disp.mapToGlobal(pos));
            return # Don't add SOAR menu if unavailable

        # Try to identify the clicked alert
        cursor = self.alerts_disp.cursorForPosition(pos);
        cursor.select(QTextCursor.SelectionType.BlockUnderCursor); # Select the paragraph under the cursor
        selected_block_text = cursor.selectedText();
        selected_alert:Optional[Dict]=None; selected_alert_index:int=-1;

        # Find the alert index based on the data-idx attribute in the HTML <p> tag
        block = cursor.block()
        if block.isValid():
            block_format = block.blockFormat()
            # Retrieve the custom property if using RichText (requires setting it during display)
            # Alternative: Parse the HTML or use a simpler matching logic
            # For simplicity, we'll try matching text content for now.
            # A better approach might involve storing QTextBlock user data.

            # Matching logic (simple): Find which alert's text is in the selected block
            match_score = {} # idx: score
            for idx, alert_data in enumerate(self._alerts_data):
                score = 0
                # Check if key elements are present in the selected block
                if qn:=alert_data.get('_query_name'):
                    if f"Source Query: {qn}" in selected_block_text: score += 2
                if ts:=alert_data.get('_time'):
                     # Convert ts to display format used earlier for better matching potential
                     # (This part is complex and might not be perfectly reliable)
                     if str(ts) in selected_block_text: score +=1
                if hst:=alert_data.get('host'):
                     if f"Host:</b> {hst}" in selected_block_text: score +=1
                if sip:=alert_data.get('src_ip'):
                     if f"Src Ip:</b> {sip}" in selected_block_text: score +=1
                if score > 0: match_score[idx] = score

            # Find best match (highest score)
            if match_score:
                best_match_idx = max(match_score, key=match_score.get)
                if match_score[best_match_idx] >= 1: # Require at least some match
                    selected_alert_index = best_match_idx
                    selected_alert = self._alerts_data[selected_alert_index]
                    logger.debug(f"Context menu: Matched alert index {selected_alert_index} with score {match_score[best_match_idx]}.")

        if not selected_alert:
            logger.warning("Context menu: Could not reliably determine the clicked alert.");
            no_alert_action = menu.addAction("Could not identify alert")
            no_alert_action.setEnabled(False)
            menu.exec(self.alerts_disp.mapToGlobal(pos));
            return

        # Add SOAR Playbook submenu
        menu.addSeparator();
        soar_submenu=menu.addMenu(QIcon.fromTheme("system-run"),"Trigger SOAR Playbook");

        playbooks_added=False;
        for pb_config in playbooks:
            pb_name=pb_config.get('name'); pb_id=pb_config.get('id'); context_fields=pb_config.get('context_fields',[]);
            if not pb_name or not pb_id: continue # Skip invalid playbook configs

            # Prepare context dictionary based on available fields in the selected alert
            playbook_context={}; context_complete=True; missing_fields=[]
            for field_name in context_fields:
                 field_value = selected_alert.get(field_name)
                 if field_value is not None: playbook_context[field_name]=field_value
                 else: context_complete=False; missing_fields.append(field_name)

            pb_action=soar_submenu.addAction(f"{pb_name}");
            if context_complete:
                # Use lambda to capture parameters at definition time
                pb_action.triggered.connect(lambda checked=False, p_id=pb_id, ctx=playbook_context.copy(), p_name=pb_name: self.trigger_soar_action_confirmed(p_id, ctx, p_name));
                tooltip_ctx = ", ".join(playbook_context.keys()) if playbook_context else "None"
                pb_action.setToolTip(f"Run '{pb_name}' (ID: {pb_id})\nContext: [{tooltip_ctx}]");
                playbooks_added=True
            else:
                pb_action.setEnabled(False);
                pb_action.setToolTip(f"Cannot run '{pb_name}': Missing required context field(s): {', '.join(missing_fields)}")

        if not playbooks_added:
            no_pbs_action = soar_submenu.addAction("No applicable playbooks found (check context)")
            no_pbs_action.setEnabled(False)

        menu.exec(self.alerts_disp.mapToGlobal(pos))

    def trigger_soar_action_confirmed(self, pb_id:str, ctx:Dict, pb_name:str):
        logger.info(f"Requesting confirmation to trigger SOAR playbook '{pb_name}' (ID:{pb_id})");
        context_summary="\nContext Provided:\n"+"\n".join([f" - {k}: {str(v)[:50]}{'...' if len(str(v))>50 else ''}" for k,v in ctx.items()]) if ctx else "\n(No context)";
        confirmation_message=f"Trigger the following SOAR Playbook?\n\n Playbook: '{pb_name}' (ID: {pb_id})\n{context_summary}"

        reply=QMessageBox.question(self,"Confirm SOAR Action",confirmation_message,
                                   QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.Cancel,
                                   QMessageBox.StandardButton.Cancel) # Default to Cancel

        if reply==QMessageBox.StandardButton.Yes:
            if self.soar and self.soar.is_configured and self.app_cfg.get('soar',{}).get('enabled'):
                # Show immediate feedback
                self.notifications.show_message(f"Triggering SOAR playbook '{pb_name}'...",level="info",duration=2000)
                QApplication.processEvents()

                def soar_task():
                    success, message = self.soar.trigger_playbook(pb_id, ctx);
                    level = "success" if success else "error";
                    final_message = f"SOAR Trigger ('{pb_name}'): {message}";
                    # Schedule result notification back on the main thread
                    QMetaObject.invokeMethod(self.notifications,"show_message",Qt.ConnectionType.QueuedConnection,
                                             Q_ARG(str,final_message),Q_ARG(int,6000),Q_ARG(str,level))

                # Run SOAR API call in a background thread
                threading.Thread(target=soar_task,daemon=True).start();
            else:
                logger.error("SOAR trigger aborted: SOAR client not available or disabled.")
                self.notifications.show_message("SOAR is not configured or enabled.",level="error")
        else:
            logger.info("SOAR playbook trigger cancelled by user.")


    @pyqtSlot(str,object)
    def update_video_frame(self, cam_name:str, frame:Any):
        video_label = self.vid_widgets.get(cam_name)
        if not video_label: return # Widget might not exist (e.g., during UI rebuild)
        if not isinstance(frame,np.ndarray) or frame.size==0:
            # logger.debug(f"Received invalid frame for {cam_name}. Type: {type(frame)}") # Too noisy
            return

        try:
            height, width, channels = frame.shape;
            bytes_per_line = channels * width;

            # Create QImage directly from the NumPy array data (expects RGB)
            qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888);

            # Create QPixmap from QImage
            pixmap = QPixmap.fromImage(qt_image);

            if pixmap.isNull(): logger.warning(f"Failed to create QPixmap for camera '{cam_name}'."); return

            # Scale pixmap to fit the label size while maintaining aspect ratio
            label_size = video_label.size();
            if label_size.isValid() and label_size.width() > 10 and label_size.height() > 10: # Check for valid size
                scaled_pixmap = pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation);
                video_label.setPixmap(scaled_pixmap)
            else: # Fallback if label size is invalid (e.g., during init)
                video_label.setPixmap(pixmap) # Show unscaled initially

            # Clear any "Initializing..." or error text if showing frame
            if video_label.text():
                video_label.setText("");
                video_label.setStyleSheet("background-color:#111;border:1px solid #444;border-radius:3px;") # Set to normal background

        except Exception as e:
            logger.error(f"Error updating video frame for '{cam_name}': {e}",exc_info=True);
            # Display error message on the video label
            video_label.setText(f"Frame Display Error\n{type(e).__name__}");
            video_label.setStyleSheet("background-color:#300;color:red;border:1px solid red;"); # Error style


    @pyqtSlot(str, bool, str)
    def on_camera_connection_status(self, cam_name:str, is_connected:bool, error_message:str):
        status_label = self.stat_labels.get(cam_name)
        video_label = self.vid_widgets.get(cam_name)
        cam_box = self.cam_boxes.get(cam_name)
        ptz_widget = self.ptz_ctrls.get(cam_name)
        camera = self.cams.get(cam_name) # Get the camera object for PTZ status

        if not status_label or not video_label:
            logger.warning(f"UI elements for camera '{cam_name}' not found during status update.")
            return

        base_title = cam_name; # Original name for the box title
        error_summary = error_message[:100]+('...'if len(error_message)>100 else '') if error_message else ""

        if is_connected:
            status_label.setText("🟢 Connected");
            status_label.setStyleSheet("font-size:8pt;color:#4CAF50;font-weight:bold;"); # Green, bold
            status_label.setToolTip("Camera is connected and streaming.");
            if cam_box: cam_box.setTitle(base_title); # Reset title if it had (Offline)

            # If video label is showing text (like "Disconnected"), clear it
            if video_label.text() and not video_label.pixmap(): # Check if text is shown without a valid pixmap
                video_label.setText("");
                video_label.setStyleSheet("background-color:#1e1e1e;color:#888;border:1px solid #444;border-radius:3px;") # Default waiting style
                video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        else: # Disconnected
            status_label.setText("🔴 Disconnected");
            status_label.setStyleSheet("font-size:8pt;color:#F44336;font-weight:bold;"); # Red, bold
            tooltip_text = "Camera disconnected." + (f"\nLast Error: {error_summary}" if error_summary else "\nAttempting to reconnect...")
            status_label.setToolTip(tooltip_text);
            if cam_box: cam_box.setTitle(f"{base_title} (Offline)");

            # Show disconnected message on video label if no pixmap is currently shown
            if not video_label.pixmap() or video_label.pixmap().isNull():
                display_text = f"Disconnected\n({error_summary if error_summary else 'Reconnecting...'})";
                video_label.setText(display_text);
                video_label.setStyleSheet("background-color:#333;color:#aaa;border:1px solid #555;border-radius:3px;"); # Disconnected style
                video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Update PTZ control visibility and enabled state
        ptz_buttons_enabled = False
        ptz_tooltip_suffix = ""
        if ptz_widget and camera and camera.is_onvif:
            ptz_widget.setVisible(True); # Show the PTZ container if ONVIF
            if is_connected and camera.ptz is not None:
                 ptz_buttons_enabled = True # Enable only if connected AND PTZ service exists
                 ptz_tooltip_suffix = ""
            elif is_connected and camera.ptz is None:
                 ptz_buttons_enabled = False
                 ptz_tooltip_suffix = " (PTZ N/A)" # Connected but PTZ failed/unavailable
            else: # Disconnected
                 ptz_buttons_enabled = False
                 ptz_tooltip_suffix = " (Offline)"

            # Update all buttons within the PTZ widget
            ptz_buttons = getattr(ptz_widget, '_ptz_buttons', [])
            for button in ptz_buttons:
                button.setEnabled(ptz_buttons_enabled)
                base_tooltip = button.toolTip().split(" (")[0] # Get base part like "Tilt Up"
                button.setToolTip(base_tooltip + ptz_tooltip_suffix)

        elif ptz_widget: # If PTZ widget exists but cam is not ONVIF
             ptz_widget.setVisible(False)


    @pyqtSlot(str)
    def on_motion_detected(self, cam_name:str):
        motion_indicator = self.mot_inds.get(cam_name)
        if motion_indicator:
            # Change indicator style to active (e.g., yellow)
            motion_indicator.setStyleSheet("background-color:#ffdd00;border:1px solid #ffaa00;border-radius:8px;");
            # Set a timer to reset the indicator style after a short duration
            QTimer.singleShot(1200, lambda name=cam_name: self._reset_motion_indicator(name)) # Use lambda to pass name

        # Optionally highlight the entire camera box
        camera_box = self.cam_boxes.get(cam_name)
        if camera_box:
            self.highlight_widget(camera_box, duration_ms=1500, color=QColor("#ffdd00")) # Highlight yellow


    def _reset_motion_indicator(self, cam_name:str):
        motion_indicator = self.mot_inds.get(cam_name)
        if motion_indicator:
            # Reset to default inactive style
            motion_indicator.setStyleSheet("background-color:transparent;border:1px solid #666;border-radius:8px;")

    @pyqtSlot(str)
    def take_snapshot(self, cam_name:str):
        logger.info(f"Snapshot requested for camera: {cam_name}");
        video_label = self.vid_widgets.get(cam_name)
        if not video_label:
            self.notifications.show_message(f"Cannot take snapshot: UI element for '{cam_name}' not found.",level="error"); return

        current_pixmap = video_label.pixmap();
        if not current_pixmap or current_pixmap.isNull():
            self.notifications.show_message(f"Snapshot failed: No valid image available for '{cam_name}'.",level="warning"); return

        # Prepare filename and default save directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S");
        safe_cam_name = "".join(c if c.isalnum() else"_" for c in cam_name).strip('_'); # Sanitize name
        default_filename = f"{safe_cam_name}_snapshot_{timestamp}.png";
        snapshot_dir = "snapshots"; # Relative directory name

        try: os.makedirs(snapshot_dir,exist_ok=True) # Ensure directory exists
        except OSError as e: logger.error(f"Error creating snapshot directory '{snapshot_dir}': {e}"); self.notifications.show_message("Snapshot Error: Could not create directory.",level="error"); return

        default_path = os.path.join(snapshot_dir, default_filename);

        # Open 'Save File' dialog
        file_path, selected_filter = QFileDialog.getSaveFileName(self,
                                        f"Save Snapshot - {cam_name}",
                                        default_path,
                                        "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;Bitmap Image (*.bmp)")

        if file_path: # User selected a path and clicked Save
            try:
                save_format = None; file_extension = os.path.splitext(file_path)[1].lower();
                # Determine format based on filter or extension
                if "PNG" in selected_filter or file_extension==".png": save_format="PNG"
                elif "JPEG" in selected_filter or file_extension in [".jpg",".jpeg"]: save_format="JPG"
                elif "Bitmap" in selected_filter or file_extension==".bmp": save_format="BMP"
                else: # Default to PNG if filter/extension is unclear
                    if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        file_path += ".png" # Append extension if missing
                    save_format = "PNG"
                    logger.warning(f"Unknown snapshot format from filter '{selected_filter}', defaulting to PNG.")

                # Save the pixmap
                if current_pixmap.save(file_path, format=save_format):
                    logger.info(f"Snapshot saved successfully: {file_path}");
                    self.notifications.show_message(f"Snapshot saved: {os.path.basename(file_path)}",level="success")
                else:
                    logger.error(f"Snapshot save failed for file: {file_path}");
                    self.notifications.show_message("Snapshot save failed (check permissions?).",level="error")
            except Exception as e:
                logger.error(f"Exception during snapshot save: {e}",exc_info=True);
                self.notifications.show_message(f"Snapshot Save Error: {e}",level="error")
        else:
             logger.info(f"Snapshot save cancelled for {cam_name}.")

    @pyqtSlot(str,float,float,float)
    def start_ptz(self, cam_name:str, pan_speed:float, tilt_speed:float, zoom_speed:float):
        logger.debug(f"PTZ Start command: Camera='{cam_name}', Pan={pan_speed:.2f}, Tilt={tilt_speed:.2f}, Zoom={zoom_speed:.2f}");
        camera = self.cams.get(cam_name);
        if camera and camera.is_connected and camera.ptz:
            camera.move_ptz(pan_speed, tilt_speed, zoom_speed)
        elif camera and not camera.is_connected:
            logger.warning(f"Cannot start PTZ for '{cam_name}': Camera disconnected.")
        elif camera and not camera.ptz:
             logger.warning(f"Cannot start PTZ for '{cam_name}': PTZ controls not available for this camera.")
        elif not camera:
             logger.error(f"Cannot start PTZ: Camera object '{cam_name}' not found.")

    @pyqtSlot(str)
    def stop_ptz(self, cam_name:str):
        logger.debug(f"PTZ Stop command: Camera='{cam_name}'");
        camera = self.cams.get(cam_name);
        if camera and camera.is_connected and camera.ptz:
            camera.stop_ptz()
        # No warning if disconnected or no PTZ, as release event often triggers stop

    # === Map View Methods === (Keep full implementations)
    def create_map_view_tab(self):
        map_tab_widget=QWidget();
        layout=QVBoxLayout(map_tab_widget);
        layout.setContentsMargins(0,0,0,0); # No margins for map view
        layout.setSpacing(0); # No spacing between toolbar and view

        # Toolbar
        toolbar=QToolBar("Map Tools");
        toolbar.setIconSize(QSize(18,18));
        toolbar.setMovable(False);
        layout.addWidget(toolbar);

        # Toolbar Actions
        load_map_action=QAction(QIcon.fromTheme("document-open"),"Load Map Image...",self);
        load_map_action.setStatusTip("Select and load a background image for the map");
        load_map_action.triggered.connect(self.select_and_load_map_image);
        toolbar.addAction(load_map_action);

        toolbar.addSeparator();

        # Edit Mode Action (Toggle Button)
        self.map_edit_mode_action=QAction(QIcon.fromTheme("document-edit"),"Edit Camera Layout",self);
        self.map_edit_mode_action.setCheckable(True);
        self.map_edit_mode_action.setChecked(self.map_edit); # Initial state from variable
        self.map_edit_mode_action.setStatusTip("Toggle mode to move camera markers on the map");
        self.map_edit_mode_action.triggered.connect(self.toggle_map_edit_mode);
        toolbar.addAction(self.map_edit_mode_action);

        toolbar.addSeparator();

        # Zoom Actions
        zoom_in_action=QAction(QIcon.fromTheme("zoom-in"),"Zoom In",self);
        zoom_in_action.setStatusTip("Zoom in on the map");
        zoom_in_action.triggered.connect(lambda: self.map_view.scale(1.2,1.2) if self.map_view else None);
        zoom_in_action.setShortcut("Ctrl++");
        toolbar.addAction(zoom_in_action);

        zoom_out_action=QAction(QIcon.fromTheme("zoom-out"),"Zoom Out",self);
        zoom_out_action.setStatusTip("Zoom out on the map");
        zoom_out_action.triggered.connect(lambda: self.map_view.scale(1/1.2,1/1.2) if self.map_view else None);
        zoom_out_action.setShortcut("Ctrl+-");
        toolbar.addAction(zoom_out_action);

        zoom_fit_action=QAction(QIcon.fromTheme("zoom-fit-best"),"Fit Map to View",self);
        zoom_fit_action.setStatusTip("Zoom to fit the entire map or markers in the view");
        zoom_fit_action.triggered.connect(self.fit_map_to_view);
        zoom_fit_action.setShortcut("Ctrl+0");
        toolbar.addAction(zoom_fit_action);

        toolbar.addSeparator();

        # Pan Mode Action (Part of an exclusive group for potential future tools)
        pan_mode_action=QAction(QIcon.fromTheme("transform-move"),"Pan Mode",self);
        pan_mode_action.setCheckable(True);
        pan_mode_action.setChecked(True); # Default mode
        pan_mode_action.setStatusTip("Enable panning the map with the mouse");
        pan_mode_action.triggered.connect(lambda checked: self.map_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag) if checked and self.map_view else None);

        # Tool group (optional, useful if adding select/other tools later)
        tool_action_group=QActionGroup(self);
        tool_action_group.addAction(pan_mode_action);
        tool_action_group.setExclusive(True);
        toolbar.addAction(pan_mode_action);

        # Graphics Scene and View
        self.map_scene=QGraphicsScene(self);
        self.map_view=QGraphicsView(self.map_scene);
        self.map_view.setRenderHints(QPainter.RenderHint.Antialiasing|QPainter.RenderHint.SmoothPixmapTransform);
        self.map_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag); # Default drag mode
        self.map_view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse); # Zoom centered on mouse
        self.map_view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter); # Keep center on resize

        # Set background colors for view and scene based on theme palette
        view_background_color=self.palette().color(QPalette.ColorRole.AlternateBase); # Slightly lighter than base
        self.map_view.setBackgroundBrush(view_background_color);
        scene_background_color=self.palette().color(QPalette.ColorRole.Base); # Base window color
        self.map_scene.setBackgroundBrush(scene_background_color);
        self.map_view.setStyleSheet("QGraphicsView { border: 1px solid #444; }"); # Add a border

        layout.addWidget(self.map_view); # Add view below toolbar

        # Initial map load and tab insertion
        self.load_map_image(); # Load image based on current config
        self.tabs.insertTab(1, map_tab_widget, QIcon.fromTheme("applications-geomap"), "Map View");
        logger.debug("Map View tab created and initialized.")

    def select_and_load_map_image(self):
        current_map_path = self.app_cfg['map_view'].get('image_path');
        start_directory = os.path.dirname(current_map_path) if current_map_path and os.path.exists(os.path.dirname(current_map_path)) else "";

        file_path, _ = QFileDialog.getOpenFileName(self,
                            "Load Map Background Image",
                            start_directory,
                            "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)");

        if file_path and current_map_path != file_path:
            logger.info(f"User selected new map image: {file_path}");
            self.app_cfg['map_view']['image_path'] = file_path;
            self.load_map_image(); # Reload the view with the new image
            self.mark_settings_dirty(); # Mark config as changed

            # Ask user if they want to save the config now
            reply = QMessageBox.question(self,"Save Configuration?",
                                         "The map background image has been changed. Save the configuration now?",
                                         QMessageBox.StandardButton.Save|QMessageBox.StandardButton.Cancel,
                                         QMessageBox.StandardButton.Save);
            if reply == QMessageBox.StandardButton.Save:
                self.save_config()
            else:
                 logger.info("Map image change not saved immediately.")
        elif file_path and current_map_path == file_path:
             logger.debug("User selected the same map image again.")
        else:
             logger.debug("Map image selection cancelled.")


    def load_map_image(self):
        if not self.map_scene or not self.map_view:
            logger.error("Cannot load map image: Map scene or view not initialized."); return;

        map_image_path = self.app_cfg['map_view'].get('image_path');
        logger.debug(f"Attempting to load map image from: {map_image_path}");

        # Remove existing background item if it exists
        if self.map_bg and self.map_bg in self.map_scene.items():
            self.map_scene.removeItem(self.map_bg);
            self.map_bg=None
            logger.debug("Removed existing map background item.")

        if map_image_path and os.path.exists(map_image_path):
            try:
                pixmap = QPixmap(map_image_path);
                if pixmap.isNull():
                    raise ValueError(f"Failed to load image, QPixmap is null: {map_image_path}");

                self.map_bg=QGraphicsPixmapItem(pixmap);
                self.map_bg.setZValue(-10); # Ensure background is behind markers
                self.map_scene.addItem(self.map_bg);
                self.map_scene.setSceneRect(self.map_bg.boundingRect()); # Set scene size to image size
                self.fit_map_to_view(); # Adjust zoom after loading
                logger.info(f"Successfully loaded map background: {map_image_path}");
                if hasattr(self,'status_bar'): self.status_bar.showMessage(f"Map loaded: {os.path.basename(map_image_path)}", 5000)
                # Also update the label in settings tab
                if hasattr(self,'map_image_path_label'): self.map_image_path_label.setText(os.path.basename(map_image_path)); self.map_image_path_label.setToolTip(map_image_path);

            except Exception as e:
                logger.error(f"Error loading map image '{map_image_path}': {e}", exc_info=True);
                if hasattr(self,'notifications'):self.notifications.show_message(f"Map Load Error: {e}",level="error");
                # Clear invalid path from config and UI
                self.app_cfg['map_view']['image_path'] = None
                if hasattr(self,'map_image_path_label'): self.map_image_path_label.setText("<i>No image / Load Error</i>"); self.map_image_path_label.setToolTip("");
                # Set a default scene rect if loading failed
                if self.map_scene.sceneRect().isEmpty(): self.map_scene.setSceneRect(QRectF(0,0,800,600)); self.fit_map_to_view()

        else: # No valid path provided or file doesn't exist
            if map_image_path: logger.warning(f"Map image file not found: {map_image_path}"); if hasattr(self,'notifications'):self.notifications.show_message(f"Map image not found: {os.path.basename(map_image_path)}",level="warning")
            else: logger.info("No map image path configured."); if hasattr(self,'status_bar'):self.status_bar.showMessage("No map image loaded.", 5000)
            # Update settings UI label
            if hasattr(self,'map_image_path_label'): self.map_image_path_label.setText("<i>No image configured</i>"); self.map_image_path_label.setToolTip("");
            # Set a default scene rect if no image and scene is empty
            if self.map_scene.sceneRect().isEmpty() and not self.map_scene.items(): self.map_scene.setSceneRect(QRectF(0,0,800,600)); self.fit_map_to_view()

        # Always update markers after potentially changing the map background/scene rect
        self.update_map_markers()


    def fit_map_to_view(self):
        if not self.map_view or not self.map_scene: return;

        # Determine the bounding rectangle to fit
        target_rect = QRectF();
        if self.map_bg:
            target_rect = self.map_bg.boundingRect() # Fit to background image if present
        else:
             target_rect = self.map_scene.itemsBoundingRect(); # Fit to all items (markers) if no background

        # If still no valid rect (empty scene), use the sceneRect itself
        if not target_rect.isValid() or target_rect.isEmpty():
             target_rect = self.map_scene.sceneRect();

        # Check if the calculated rectangle is valid before fitting
        if target_rect.isValid() and not target_rect.isEmpty() and target_rect.width()>0 and target_rect.height()>0:
            # Add a small margin around the target rectangle for better viewing
            margin_x=target_rect.width()*0.05; margin_y=target_rect.height()*0.05;
            rect_with_margin=target_rect.adjusted(-margin_x,-margin_y,margin_x,margin_y);
            self.map_view.fitInView(rect_with_margin, Qt.AspectRatioMode.KeepAspectRatio);
            logger.debug(f"Fitted map view to rectangle: {target_rect}")
        else:
            logger.debug("Fit map to view skipped: Target rectangle is invalid or empty.")


    def update_map_markers(self):
        if not self.map_scene or not self._def_cam_icon:
            logger.warning("Cannot update map markers: Scene or default icon not available.")
            return;

        logger.debug("Updating map markers...");
        configured_cameras = {c.get('name') for c in self.app_cfg.get('cameras',[]) if c.get('name')}
        saved_positions = self.app_cfg['map_view'].get('camera_positions',{});
        existing_marker_names = set(self.map_markers.keys());

        # Remove markers for cameras that no longer exist in config
        markers_to_remove = existing_marker_names - configured_cameras
        for name in markers_to_remove:
            marker = self.map_markers.pop(name,None);
            if marker and marker.scene() == self.map_scene:
                self.map_scene.removeItem(marker);
                logger.debug(f"Removed map marker for deleted camera: {name}")

        # Add or update markers for configured cameras
        for cam_config in self.app_cfg.get('cameras',[]):
            name=cam_config.get('name');
            if not name: continue # Should not happen if validation is correct

            marker = self.map_markers.get(name);
            position_data = saved_positions.get(name);
            target_position: Optional[QPointF] = None

            # Parse position data
            if isinstance(position_data,dict) and 'x' in position_data and 'y' in position_data:
                 try: target_position=QPointF(float(position_data['x']), float(position_data['y']))
                 except (ValueError, TypeError): logger.warning(f"Invalid position data for camera '{name}'."); target_position=None

            if marker: # Marker exists, update its position if needed
                marker.setEditMode(self.map_edit); # Ensure edit mode is current
                if target_position is not None and marker.pos() != target_position:
                    logger.debug(f"Updating position for existing marker '{name}' to {target_position}");
                    marker.setPos(target_position);
                # Ensure it's in the scene (might have been removed incorrectly)
                if marker.scene() != self.map_scene:
                     self.map_scene.addItem(marker)
            elif target_position is not None: # Marker doesn't exist but has a position, create it
                try:
                    logger.debug(f"Creating new map marker for '{name}' at {target_position}");
                    new_marker=CameraMarkerItem(name,self._def_cam_icon.copy(),target_position);
                    new_marker.setEditMode(self.map_edit);
                    # Connect signals
                    new_marker.markerClicked.connect(self.on_marker_clicked);
                    new_marker.markerMoved.connect(self.on_marker_moved);
                    self.map_scene.addItem(new_marker);
                    self.map_markers[name]=new_marker;
                except Exception as e: logger.error(f"Failed to create map marker for '{name}': {e}",exc_info=True)
            # else: marker doesn't exist and has no position -> do nothing until placed in edit mode

        # If in edit mode, add markers for cameras without positions
        if self.map_edit:
            self._add_markers_for_unplaced_cameras()

        logger.debug(f"Map markers update complete. Current marker count: {len(self.map_markers)}")


    def toggle_map_edit_mode(self, checked:bool):
        if self.map_edit == checked: return; # No change
        self.map_edit = checked;
        logger.info(f"Map Edit Mode toggled: {'ON' if checked else 'OFF'}");

        if hasattr(self,'status_bar'): self.status_bar.showMessage(f"Map Edit Mode: {'ENABLED' if checked else 'DISABLED'}", 3000);

        # Update edit state for all existing markers
        for marker in self.map_markers.values():
            marker.setEditMode(self.map_edit)

        if checked:
            # Add markers for any cameras that don't have a position yet
            self._add_markers_for_unplaced_cameras()
            # Set drag mode to NoDrag to allow item moving, keep pan action unchecked
            if self.map_view: self.map_view.setDragMode(QGraphicsView.DragMode.NoDrag)
            # Consider finding the 'Pan Mode' action and unchecking it if necessary
        else: # Edit mode turned OFF
             # Restore Pan drag mode
             if self.map_view: self.map_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
             # Consider finding the 'Pan Mode' action and checking it if necessary

             # Check for unsaved changes (moved markers)
             if self._dirty:
                 reply=QMessageBox.question(self,"Save Map Changes?",
                                           "You have moved camera markers. Save the new positions?",
                                            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                                            QMessageBox.StandardButton.Save) # Default to Save
                 if reply == QMessageBox.StandardButton.Save:
                     if not self.save_config(): # Attempt save
                         # If save fails, revert edit mode toggle
                         logger.warning("Save failed. Reverting map edit mode toggle.")
                         self.map_edit=True; self.map_edit_mode_action.setChecked(True);
                         if hasattr(self,'status_bar'): self.status_bar.showMessage("Save failed. Map Edit Mode remains ENABLED.", 4000);
                         [m.setEditMode(True) for m in self.map_markers.values()] # Re-enable edit on markers
                         if self.map_view: self.map_view.setDragMode(QGraphicsView.DragMode.NoDrag) # Set drag mode back
                 elif reply == QMessageBox.StandardButton.Cancel:
                      # User cancelled, revert the toggle action
                      self.map_edit=True; self.map_edit_mode_action.setChecked(True); # Set back to checked
                      logger.info("Map Edit Mode toggle cancelled by user.")
                      # No need to change marker edit mode or view drag mode, they are still in edit state
                 elif reply == QMessageBox.StandardButton.Discard:
                      logger.info("Discarding map position changes.")
                      # Reload positions from config to revert visual changes (optional but good)
                      self.load_map_image() # This should reload markers with saved positions
                      self.mark_settings_dirty(False) # Discarded changes, so not dirty anymore


    def _add_markers_for_unplaced_cameras(self):
        if not self.map_edit: return # Only run when entering edit mode
        logger.debug("Checking for unplaced cameras to add markers...");
        if not self.map_scene or not self._def_cam_icon: return;

        configured_cameras = {c.get('name') for c in self.app_cfg.get('cameras',[]) if c.get('name')}
        placed_cameras = set(self.map_markers.keys())
        unplaced_cameras = configured_cameras - placed_cameras

        if not unplaced_cameras: logger.debug("No unplaced cameras found."); return;

        logger.info(f"Adding temporary markers for {len(unplaced_cameras)} unplaced camera(s): {', '.join(unplaced_cameras)}");
        scene_rect = self.map_scene.sceneRect();
        center_point = scene_rect.center() if scene_rect.isValid() else QPointF(50,50); # Default center if no scene rect
        # Calculate initial placement positions (e.g., in a circle around center)
        radius = min(scene_rect.width(), scene_rect.height()) * 0.1 if scene_rect.isValid() and scene_rect.width()>0 and scene_rect.height()>0 else 40
        num_unplaced = len(unplaced_cameras)
        angle_step = 360.0 / num_unplaced if num_unplaced > 0 else 0

        for i, cam_name in enumerate(unplaced_cameras):
            angle_rad = i*angle_step*(np.pi/180.0);
            # Calculate initial position relative to center
            initial_pos = center_point + QPointF(radius*np.cos(angle_rad), radius*np.sin(angle_rad))

            # Constrain initial position within scene bounds using itemChange logic temporarily
            try:
                temp_marker=CameraMarkerItem(cam_name,self._def_cam_icon,QPointF(0,0));
                # Temporarily assign scene to allow itemChange boundary check
                temp_marker.scene = lambda:self.map_scene;
                constrained_pos = temp_marker.itemChange(QGraphicsItem.GraphicsItemChange.ItemPositionChange, initial_pos);
                del temp_marker # Clean up temporary object
            except Exception as e_constrain:
                 logger.warning(f"Could not pre-constrain position for '{cam_name}', using calculated: {e_constrain}")
                 constrained_pos = initial_pos

            logger.debug(f"Placing new marker for '{cam_name}' initially near {constrained_pos}")
            try:
                marker=CameraMarkerItem(name=cam_name, icon=self._def_cam_icon.copy(), pos=constrained_pos);
                marker.setEditMode(True); # Enable dragging immediately
                marker.markerClicked.connect(self.on_marker_clicked);
                marker.markerMoved.connect(self.on_marker_moved);
                self.map_scene.addItem(marker);
                self.map_markers[cam_name]=marker;
                # Trigger markerMoved immediately to save this initial position to config buffer
                self.on_marker_moved(cam_name, constrained_pos)
            except Exception as e: logger.error(f"Failed to create initial marker for unplaced camera '{cam_name}': {e}",exc_info=True)

        self.mark_settings_dirty() # Adding markers counts as a change

    def on_marker_moved(self, cam_name:str, new_pos:QPointF):
         if cam_name in self.map_markers:
              logger.info(f"Map marker '{cam_name}' moved to ({new_pos.x():.1f}, {new_pos.y():.1f})")
              # Ensure map_view structure exists
              if 'map_view' not in self.app_cfg: self.app_cfg['map_view']={}
              if 'camera_positions' not in self.app_cfg['map_view'] or not isinstance(self.app_cfg['map_view']['camera_positions'], dict):
                   self.app_cfg['map_view']['camera_positions']={}

              # Update position in the config dictionary (rounded)
              rounded_position={'x':round(new_pos.x(),2),'y':round(new_pos.y(),2)}
              if self.app_cfg['map_view']['camera_positions'].get(cam_name) != rounded_position:
                   self.app_cfg['map_view']['camera_positions'][cam_name]=rounded_position;
                   self.mark_settings_dirty() # Mark config as modified
         else: logger.warning(f"Received markerMoved signal for unknown camera: {cam_name}")

    def on_marker_clicked(self, cam_name:str):
        if self.map_edit:
             logger.debug(f"Marker '{cam_name}' clicked while in Edit Mode. Action ignored.")
             return; # Ignore clicks when in edit mode (move takes precedence)

        logger.info(f"Map marker clicked: '{cam_name}'. Switching to Monitoring tab and focusing camera.");
        monitor_tab_index=-1;
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i)=="Monitoring": monitor_tab_index=i; break

        if monitor_tab_index==-1:
            logger.warning("Cannot switch to Monitoring tab: Tab not found."); return;

        # Switch to the Monitoring tab
        self.tabs.setCurrentIndex(monitor_tab_index)

        # Find the corresponding camera box and scroll to it
        cam_box = self.cam_boxes.get(cam_name)
        if cam_box:
            monitor_tab_widget = self.tabs.widget(monitor_tab_index);
            scroll_area = monitor_tab_widget.findChild(QScrollArea); # Find the scroll area in the tab
            if scroll_area:
                 scroll_area.ensureWidgetVisible(cam_box, yMargin=50); # Scroll to make the box visible
                 QApplication.processEvents() # Process events to ensure scrolling happens
                 # Highlight the camera box visually
                 self.highlight_widget(cam_box, duration_ms=1200, color=QColor(42,130,218)) # Highlight blue
            else: logger.warning(f"Could not find ScrollArea in Monitoring tab to focus camera '{cam_name}'.")
        else: logger.warning(f"Could not find camera GroupBox for '{cam_name}' in Monitoring tab.")


    def highlight_widget(self, widget:QWidget, duration_ms:int=1500, color:QColor=QColor("#ffdd00")):
        if not widget: return;

        property_name=b"_highlight_animation" # Use bytes for property name
        original_stylesheet=widget.styleSheet() # Store original style

        # Stop any existing highlight animation on this widget
        existing_animation = widget.property(property_name)
        if existing_animation and isinstance(existing_animation,QVariantAnimation):
            logger.debug(f"Stopping existing highlight on {widget.objectName()}")
            existing_animation.stop()
            # Don't necessarily restore stylesheet here, let the new animation take over

        # Determine the border color to transition back to (try parsing from original)
        end_color = QColor("#666") # Default dark border
        try:
            style_parts = original_stylesheet.split('border:')
            if len(style_parts) > 1:
                border_prop = style_parts[1].split(';')[0].strip() # e.g., "1px solid #aabbcc"
                color_part = border_prop.split()[-1] # Get the last part, likely the color
                temp_color = QColor(color_part)
                if temp_color.isValid(): end_color = temp_color
        except Exception as e_parse: logger.warning(f"Could not parse original border color for highlight: {e_parse}")

        # Create the animation
        animation=QVariantAnimation(widget);
        widget.setProperty(property_name, animation); # Store animation on the widget
        animation.setDuration(duration_ms);
        animation.setStartValue(color); # Start with highlight color
        animation.setEndValue(end_color); # End with original/default border color
        animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

        # Define update function (closure to capture widget and original style parts)
        def update_style(current_color):
            # Reconstruct stylesheet, only changing the border color part
            # This is safer than replacing the entire stylesheet if it's complex
            base_style = original_stylesheet.split('border:')[0] if 'border:' in original_stylesheet else original_stylesheet
            border_style = f"border: 1px solid {current_color.name()};" # Assuming 1px solid, adjust if needed
            other_parts = ""
            if 'border:' in original_stylesheet:
                 try: other_parts = ';'.join(original_stylesheet.split('border:')[1].split(';')[1:])
                 except IndexError: pass # No parts after border

            new_style = base_style + border_style + (';'+other_parts if other_parts else "")

            # Special handling for QGroupBox to maintain title style etc.
            if isinstance(widget, QGroupBox):
                # More robustly preserve existing properties if possible
                title_style = ""
                if "QGroupBox::title" in original_stylesheet:
                    try: title_style = "QGroupBox::title" + original_stylesheet.split("QGroupBox::title")[1].split("}")[0] + "}"
                    except: pass
                new_style = f"QGroupBox{{{border_style} border-radius:6px;margin-top:0.6em;padding:0.8em 0.5em 0.5em 0.5em;{'font-weight:bold;' if 'font-weight:bold' in original_stylesheet else ''}}} {title_style}"

            try: widget.setStyleSheet(new_style)
            except Exception as e_apply: logger.error(f"Error applying highlight style: {e_apply}"); animation.stop() # Stop animation if style fails

        # Connect valueChanged signal to the update function
        animation.valueChanged.connect(update_style);

        # Connect finished signal to restore original style and clean up
        animation.finished.connect(lambda w=widget, orig_ss=original_stylesheet, prop=property_name: self._on_highlight_finished(w, orig_ss, prop))

        # Start the animation (will be deleted automatically when stopped/finished)
        animation.start(QVariantAnimation.DeletionPolicy.DeleteWhenStopped)


    def _on_highlight_finished(self, widget, original_stylesheet, property_name):
         if widget:
             logger.debug(f"Highlight finished for {widget.objectName()}. Restoring style.")
             widget.setStyleSheet(original_stylesheet);
             widget.setProperty(property_name, None) # Remove animation property


    # ==================== Settings Tab Methods ====================
    def create_settings_tab(self):
        settings_tab_widget=QWidget();
        main_settings_layout=QVBoxLayout(settings_tab_widget);
        main_settings_layout.setSpacing(15);
        main_settings_layout.setContentsMargins(10,10,10,10);

        # Sub-tabs for different setting categories
        settings_sub_tabs=QTabWidget()

        # --- Camera Settings Tab ---
        camera_settings_widget=QWidget(); cam_layout=QVBoxLayout(camera_settings_widget);
        cam_groupbox=QGroupBox("Camera Sources"); cam_group_layout=QHBoxLayout(cam_groupbox);
        self.camera_list_widget=QListWidget();
        self.camera_list_widget.setToolTip("Double-click a camera to edit its configuration.");
        self.camera_list_widget.itemDoubleClicked.connect(self.edit_camera_config); # Connect double-click
        self.camera_list_widget.setAlternatingRowColors(True);
        cam_group_layout.addWidget(self.camera_list_widget, stretch=1); # List takes available space
        # Buttons VBox
        cam_button_layout=QVBoxLayout(); cam_button_layout.setSpacing(8);
        cam_button_layout.addWidget(QPushButton(QIcon.fromTheme("list-add")," Add Camera...",clicked=self.add_camera_config));
        cam_button_layout.addWidget(QPushButton(QIcon.fromTheme("document-edit")," Edit Selected...",clicked=self.edit_camera_config));
        cam_button_layout.addWidget(QPushButton(QIcon.fromTheme("list-remove")," Remove Selected",clicked=self.remove_camera_config));
        cam_button_layout.addStretch(); # Push buttons up
        cam_group_layout.addLayout(cam_button_layout);
        cam_layout.addWidget(cam_groupbox)
        settings_sub_tabs.addTab(camera_settings_widget,QIcon.fromTheme("camera-video"),"Cameras");

        # --- SIEM Settings Tab ---
        siem_settings_widget=QWidget(); siem_layout=QVBoxLayout(siem_settings_widget);
        # Connection Group
        siem_conn_groupbox=QGroupBox("SIEM Connection Details"); siem_conn_form=QFormLayout(siem_conn_groupbox);
        self.siem_url_input=QLineEdit(); self.siem_url_input.setPlaceholderText("e.g., https://splunk.example.com:8089")
        self.siem_token_input=QLineEdit(); self.siem_token_input.setEchoMode(QLineEdit.EchoMode.Password);
        self.siem_auth_combo=QComboBox(); self.siem_auth_combo.addItems(["Bearer","Splunk"]); self.siem_auth_combo.setToolTip("Authorization header type (Bearer for tokens, Splunk for Splunk tokens)")
        self.siem_verify_ssl_check=QCheckBox("Verify SSL Certificate"); self.siem_verify_ssl_check.setChecked(False) # Default secure practice is True, but often internal systems use self-signed
        self.siem_refresh_input=QLineEdit(); self.siem_refresh_input.setPlaceholderText("e.g., 15"); self.siem_refresh_input.setToolTip("Interval in minutes to automatically fetch SIEM events (0 to disable).")
        siem_conn_form.addRow("API URL:",self.siem_url_input);
        siem_conn_form.addRow("Auth Type:",self.siem_auth_combo);
        siem_conn_form.addRow("API Token:",self.siem_token_input);
        siem_conn_form.addRow("", self.siem_verify_ssl_check); # Checkbox doesn't need a label on left
        siem_conn_form.addRow("Refresh Interval (min):",self.siem_refresh_input);
        siem_layout.addWidget(siem_conn_groupbox)
        # Queries Group
        siem_query_groupbox=QGroupBox("SIEM Queries"); siem_query_vlayout=QVBoxLayout(siem_query_groupbox); siem_query_vlayout.setSpacing(5);
        self.siem_query_table=QTableWidget();
        self.siem_query_table.setColumnCount(3);
        self.siem_query_table.setHorizontalHeaderLabels(["Enabled","Query Name","Search Query (SPL)"]);
        self.siem_query_table.horizontalHeader().setStretchLastSection(True); # Query column takes extra space
        self.siem_query_table.verticalHeader().setVisible(False); # No row numbers needed
        self.siem_query_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows); # Select whole row
        self.siem_query_table.setAlternatingRowColors(True);
        self.siem_query_table.setColumnWidth(0, 60); # Width for checkbox column
        self.siem_query_table.setColumnWidth(1, 150); # Width for name column
        self.siem_query_table.itemChanged.connect(self.siem_query_item_changed) # Connect checkbox changes
        # Query Buttons HBox
        siem_query_button_hlayout=QHBoxLayout();
        siem_query_button_hlayout.addWidget(QPushButton(QIcon.fromTheme("list-add")," Add Query...",clicked=self.add_siem_query));
        siem_query_button_hlayout.addWidget(QPushButton(QIcon.fromTheme("document-edit")," Edit Selected...",clicked=self.edit_siem_query));
        siem_query_button_hlayout.addWidget(QPushButton(QIcon.fromTheme("list-remove")," Remove Selected",clicked=self.remove_siem_query));
        siem_query_button_hlayout.addStretch(); # Push buttons left
        siem_query_vlayout.addWidget(self.siem_query_table); # Add table
        siem_query_vlayout.addLayout(siem_query_button_hlayout); # Add button row
        siem_layout.addWidget(siem_query_groupbox)
        settings_sub_tabs.addTab(siem_settings_widget,QIcon.fromTheme("utilities-log"),"SIEM Fetch");

        # --- SOAR Settings Tab ---
        soar_settings_widget=QWidget(); soar_layout=QVBoxLayout(soar_settings_widget);
        # Connection Group
        soar_conn_groupbox=QGroupBox("SOAR Connection Details"); soar_conn_form=QFormLayout(soar_conn_groupbox);
        self.soar_enabled_check=QCheckBox("Enable SOAR Integration"); self.soar_enabled_check.setToolTip("Enable triggering of SOAR playbooks from SIEM events.")
        self.soar_api_url_input=QLineEdit(); self.soar_api_url_input.setPlaceholderText("e.g., https://soar.example.com")
        self.soar_api_key_input=QLineEdit(); self.soar_api_key_input.setEchoMode(QLineEdit.EchoMode.Password);
        self.soar_auth_header_name_input=QLineEdit("Authorization"); self.soar_auth_header_name_input.setToolTip("Name of the HTTP header for the API key (e.g., Authorization, ph-auth-token)")
        self.soar_auth_header_prefix_input=QLineEdit("Bearer "); self.soar_auth_header_prefix_input.setToolTip("Prefix before the API key in the header (e.g., 'Bearer ', 'Token ', leave empty if none)")
        self.soar_verify_ssl_check=QCheckBox("Verify SSL Certificate"); self.soar_verify_ssl_check.setChecked(True); # Default True for external systems
        soar_conn_form.addRow("", self.soar_enabled_check);
        soar_conn_form.addRow("SOAR API URL:",self.soar_api_url_input);
        soar_conn_form.addRow("SOAR API Key:",self.soar_api_key_input);
        soar_conn_form.addRow("Auth Header Name:",self.soar_auth_header_name_input);
        soar_conn_form.addRow("Auth Header Prefix:",self.soar_auth_header_prefix_input);
        soar_conn_form.addRow("", self.soar_verify_ssl_check);
        soar_layout.addWidget(soar_conn_groupbox)
        # Playbooks Group
        soar_playbook_groupbox=QGroupBox("SOAR Playbooks"); soar_pb_vlayout=QVBoxLayout(soar_playbook_groupbox);
        self.soar_playbook_table=QTableWidget();
        self.soar_playbook_table.setColumnCount(3);
        self.soar_playbook_table.setHorizontalHeaderLabels(["Playbook Name","Playbook ID / Identifier","Required Context Fields (CSV)"]);
        self.soar_playbook_table.horizontalHeader().setStretchLastSection(True); # Context field list can be long
        self.soar_playbook_table.verticalHeader().setVisible(False);
        self.soar_playbook_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows);
        self.soar_playbook_table.setAlternatingRowColors(True);
        self.soar_playbook_table.setColumnWidth(0, 180) # Playbook name
        self.soar_playbook_table.setColumnWidth(1, 150) # Playbook ID
        self.soar_playbook_table.setToolTip("Define playbooks triggerable from SIEM events. Context fields must exist in the SIEM event.")
        # Playbook Buttons HBox
        soar_pb_button_hlayout=QHBoxLayout();
        soar_pb_button_hlayout.addWidget(QPushButton(QIcon.fromTheme("list-add")," Add Playbook...",clicked=self.add_soar_playbook));
        soar_pb_button_hlayout.addWidget(QPushButton(QIcon.fromTheme("document-edit")," Edit Selected...",clicked=self.edit_soar_playbook));
        soar_pb_button_hlayout.addWidget(QPushButton(QIcon.fromTheme("list-remove")," Remove Selected",clicked=self.remove_soar_playbook));
        soar_pb_button_hlayout.addStretch();
        soar_pb_vlayout.addWidget(self.soar_playbook_table);
        soar_pb_vlayout.addLayout(soar_pb_button_hlayout);
        soar_layout.addWidget(soar_playbook_groupbox)
        settings_sub_tabs.addTab(soar_settings_widget,QIcon.fromTheme("system-run"),"SOAR Actions");

        # --- Map View Settings Tab ---
        map_settings_widget=QWidget(); map_layout=QVBoxLayout(map_settings_widget);
        map_image_groupbox=QGroupBox("Map Background Image"); map_image_hlayout=QHBoxLayout(map_image_groupbox);
        self.map_image_path_label=QLabel("<i>No image selected</i>");
        self.map_image_path_label.setWordWrap(True); # Allow long paths to wrap
        load_map_button=QPushButton(QIcon.fromTheme("document-open")," Select Map Image...");
        load_map_button.clicked.connect(self.select_and_load_map_image); # Re-use the map tab's function
        map_image_hlayout.addWidget(self.map_image_path_label, stretch=1); # Label takes space
        map_image_hlayout.addWidget(load_map_button);
        map_layout.addWidget(map_image_groupbox);
        map_layout.addStretch(); # Push groupbox up
        settings_sub_tabs.addTab(map_settings_widget,QIcon.fromTheme("applications-geomap"),"Map View");

        # Add the sub-tabs widget to the main settings layout
        main_settings_layout.addWidget(settings_sub_tabs);

        # Apply & Save Button
        apply_save_button=QPushButton(QIcon.fromTheme("document-save")," Apply && Save All Settings");
        apply_save_button.setToolTip("Apply all changes from Settings tabs and save the configuration file");
        apply_save_button.setFixedHeight(35); # Make button prominent
        apply_save_button.clicked.connect(self.apply_and_save_settings);
        main_settings_layout.addWidget(apply_save_button, alignment=Qt.AlignmentFlag.AlignRight) # Align bottom-right

        # Load initial data into the UI
        self.refresh_settings_ui();

        # Add the main settings tab to the application's main tab widget
        self.tabs.addTab(settings_tab_widget, QIcon.fromTheme("preferences-system"), "Settings");
        logger.debug("Settings tab created and populated.")

    def refresh_settings_ui(self):
        logger.debug("Refreshing settings UI with current configuration...");
        # --- Cameras ---
        self.camera_list_widget.clear(); configured_names=set();
        for camera_config in self.app_cfg.get('cameras',[]):
             name=camera_config.get('name');
             if name:
                 icon_name = "camera-video" if camera_config.get('onvif') else "network-wired"
                 list_item=QListWidgetItem(f" {name}"); # Add space for icon
                 list_item.setIcon(QIcon.fromTheme(icon_name));
                 list_item.setData(Qt.ItemDataRole.UserRole,camera_config); # Store full config dict in the item
                 self.camera_list_widget.addItem(list_item);
                 configured_names.add(name)

        # --- SIEM ---
        siem_config=self.app_cfg.get('siem',{});
        self.siem_url_input.setText(siem_config.get('api_url',''));
        self.siem_token_input.setText(siem_config.get('token',''));
        auth_type=siem_config.get('auth_header','Bearer');
        combo_index=self.siem_auth_combo.findText(auth_type,Qt.MatchFlag.MatchFixedString);
        self.siem_auth_combo.setCurrentIndex(combo_index if combo_index>=0 else 0); # Default to Bearer if not found
        self.siem_verify_ssl_check.setChecked(bool(siem_config.get('verify_ssl',False)));
        self.siem_refresh_input.setText(str(siem_config.get('refresh_interval_min',15)))
        # Queries Table
        self.siem_query_table.blockSignals(True); # Block signals during programmatic update
        self.siem_query_table.setRowCount(0); # Clear table
        siem_queries=siem_config.get('queries',[]);
        self.siem_query_table.setRowCount(len(siem_queries))
        for row, query_data in enumerate(siem_queries):
             # Enabled Checkbox (Column 0)
             enabled_item=QTableWidgetItem();
             enabled_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable|Qt.ItemFlag.ItemIsEnabled); # Checkable and enabled
             enabled_item.setCheckState(Qt.CheckState.Checked if query_data.get('enabled',True) else Qt.CheckState.Unchecked);
             self.siem_query_table.setItem(row,0,enabled_item);
             # Name Item (Column 1)
             name_item=QTableWidgetItem(query_data.get('name',''));
             name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable); # Read-only in table
             self.siem_query_table.setItem(row,1,name_item);
             # Query Item (Column 2)
             query_item=QTableWidgetItem(query_data.get('query',''));
             query_item.setFlags(query_item.flags() & ~Qt.ItemFlag.ItemIsEditable); # Read-only
             query_item.setToolTip(query_data.get('query','')) # Show full query on hover
             self.siem_query_table.setItem(row,2,query_item)
        self.siem_query_table.resizeRowsToContents();
        self.siem_query_table.blockSignals(False); # Re-enable signals

        # --- SOAR ---
        soar_config=self.app_cfg.get('soar',{});
        self.soar_enabled_check.setChecked(bool(soar_config.get('enabled',False)));
        self.soar_api_url_input.setText(soar_config.get('api_url',''));
        self.soar_api_key_input.setText(soar_config.get('api_key',''));
        self.soar_auth_header_name_input.setText(soar_config.get('auth_header_name','Authorization'));
        self.soar_auth_header_prefix_input.setText(soar_config.get('auth_header_prefix','Bearer '));
        self.soar_verify_ssl_check.setChecked(bool(soar_config.get('verify_ssl',True)))
        # Playbooks Table
        self.soar_playbook_table.setRowCount(0); # Clear table
        soar_playbooks=soar_config.get('playbooks',[]);
        self.soar_playbook_table.setRowCount(len(soar_playbooks))
        for row, playbook_data in enumerate(soar_playbooks):
             # Name (Col 0), ID (Col 1), Context (Col 2)
             self.soar_playbook_table.setItem(row,0,QTableWidgetItem(playbook_data.get('name','')))
             self.soar_playbook_table.setItem(row,1,QTableWidgetItem(playbook_data.get('id','')))
             context_str = ", ".join(playbook_data.get('context_fields',[]))
             context_item = QTableWidgetItem(context_str)
             context_item.setToolTip(context_str) # Show full list on hover
             self.soar_playbook_table.setItem(row,2,context_item)
             # Make all cells read-only in the table
             for col in range(3):
                 item=self.soar_playbook_table.item(row,col);
                 if item: item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.soar_playbook_table.resizeColumnsToContents();
        self.soar_playbook_table.resizeRowsToContents()

        # --- Map ---
        map_config=self.app_cfg.get('map_view',{});
        image_path=map_config.get('image_path');
        if image_path and os.path.exists(image_path):
             self.map_image_path_label.setText(os.path.basename(image_path));
             self.map_image_path_label.setToolTip(image_path)
        else:
             self.map_image_path_label.setText("<i>No image selected or path invalid</i>");
             self.map_image_path_label.setToolTip("")
        # Clean up potentially stale camera positions (already done in _validate_config, but double-check)
        map_positions=map_config.get('camera_positions',{});
        stale_positions=set(map_positions.keys())-configured_names;
        if stale_positions:
             logger.info(f"Refreshing settings UI: Pruning stale map positions for: {stale_positions}");
             for name in stale_positions: map_positions.pop(name,None)
             self.mark_settings_dirty() # Pruning is a change

        logger.debug("Settings UI refresh complete.")


    # --- Settings CRUD Helpers ---
    def add_camera_config(self):
        dialog=CameraConfigDialog(self); # Create new dialog instance
        if dialog.exec()==QDialog.DialogCode.Accepted:
            new_config = dialog.get_config();
            if new_config: # get_config returns None on validation error
                new_name = new_config['name'];
                # Check for duplicate names
                existing_names = {c.get('name') for c in self.app_cfg['cameras']}
                if new_name in existing_names:
                     QMessageBox.warning(self,"Duplicate Name",f"A camera with the name '{new_name}' already exists. Please choose a unique name.");
                     return # Stay in settings to allow correction

                logger.info(f"Adding new camera configuration: {new_name}");
                self.app_cfg['cameras'].append(new_config);
                self.refresh_settings_ui(); # Update the list widget
                self.mark_settings_dirty();
                self.notifications.show_message(f"Camera '{new_name}' added. Remember to Apply & Save.",level="info")

    def edit_camera_config(self):
        selected_items=self.camera_list_widget.selectedItems();
        current_item=self.camera_list_widget.currentItem(); # Fallback if selection is weird

        if not selected_items and not current_item:
            QMessageBox.information(self,"Edit Camera","Please select a camera from the list to edit."); return

        item_to_edit = selected_items[0] if selected_items else current_item;
        original_config=item_to_edit.data(Qt.ItemDataRole.UserRole); # Get stored config

        if not isinstance(original_config,dict): logger.error("Invalid data found in camera list item."); return;

        original_name=original_config.get('name');
        # Use deepcopy to avoid modifying the original dict if the dialog is cancelled
        dialog=CameraConfigDialog(self, config=copy.deepcopy(original_config))

        if dialog.exec()==QDialog.DialogCode.Accepted:
            updated_config = dialog.get_config();
            if updated_config:
                updated_name=updated_config.get('name');
                # Check for duplicate names ONLY if the name was changed
                if original_name != updated_name:
                    existing_names = {c.get('name') for c in self.app_cfg['cameras'] if c.get('name') != original_name}
                    if updated_name in existing_names:
                        QMessageBox.warning(self,"Duplicate Name",f"Another camera with the name '{updated_name}' already exists. Please choose a unique name.");
                        return

                # Find the index of the original config to replace it
                index_to_update = -1;
                for i, cfg in enumerate(self.app_cfg['cameras']):
                     if cfg.get('name') == original_name: index_to_update = i; break

                if index_to_update != -1:
                    logger.info(f"Updating camera configuration: '{original_name}' -> '{updated_name}'");
                    self.app_cfg['cameras'][index_to_update] = updated_config;
                    # Handle potential rename in map positions
                    if original_name != updated_name:
                         self._handle_camera_rename_in_map(original_name,updated_name);
                    self.refresh_settings_ui(); # Update list widget
                    self.mark_settings_dirty();
                    self.notifications.show_message(f"Camera '{updated_name}' updated. Remember to Apply & Save.",level="info")
                else:
                     logger.error(f"Could not find original camera '{original_name}' in config list during edit.")
                     QMessageBox.critical(self,"Error","Could not find the camera to update in the configuration. Please reload.")


    def remove_camera_config(self):
        selected_items=self.camera_list_widget.selectedItems();
        current_item=self.camera_list_widget.currentItem();

        if not selected_items and not current_item:
            QMessageBox.information(self,"Remove Camera","Please select a camera from the list to remove."); return

        item_to_remove = selected_items[0] if selected_items else current_item;
        config_to_remove=item_to_remove.data(Qt.ItemDataRole.UserRole);
        name_to_remove=config_to_remove.get('name') if isinstance(config_to_remove,dict) else None;

        if not name_to_remove: logger.error("Invalid data in camera list item for removal."); return;

        reply = QMessageBox.question(self,"Confirm Removal",f"Are you sure you want to remove the camera '{name_to_remove}'?",
                                     QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No) # Default to No

        if reply==QMessageBox.StandardButton.Yes:
            logger.info(f"Removing camera configuration: {name_to_remove}");
            initial_length = len(self.app_cfg['cameras']);
            # Filter out the camera by name
            self.app_cfg['cameras']=[c for c in self.app_cfg['cameras'] if c.get('name')!=name_to_remove];

            if len(self.app_cfg['cameras']) < initial_length:
                self._handle_camera_remove_from_map(name_to_remove); # Remove associated map position
                self.refresh_settings_ui(); # Update list widget
                self.mark_settings_dirty();
                self.notifications.show_message(f"Camera '{name_to_remove}' removed. Remember to Apply & Save.",level="info")
            else:
                logger.error(f"Error removing camera '{name_to_remove}': Camera not found in config list after filtering.")
                QMessageBox.critical(self,"Error","Could not find the camera to remove in the configuration.")


    def _handle_camera_rename_in_map(self, old_name:str, new_name:str):
        if 'map_view' in self.app_cfg and isinstance(positions:=self.app_cfg['map_view'].get('camera_positions',{}),dict):
            if old_name in positions:
                 positions[new_name] = positions.pop(old_name); # Rename the key
                 logger.debug(f"Renamed map position key: '{old_name}' -> '{new_name}'.");
                 self.mark_settings_dirty() # Renaming is a change

    def _handle_camera_remove_from_map(self, name_to_remove:str):
        if 'map_view' in self.app_cfg and isinstance(positions:=self.app_cfg['map_view'].get('camera_positions',{}),dict):
             if name_to_remove in positions:
                 positions.pop(name_to_remove); # Remove the key
                 logger.debug(f"Removed map position for deleted camera: '{name_to_remove}'.");
                 self.mark_settings_dirty() # Removal is a change

    @pyqtSlot(QTableWidgetItem)
    def siem_query_item_changed(self, item:QTableWidgetItem):
         # Only react to changes in the "Enabled" column (column 0)
         if item.column()==0:
             row_index=item.row();
             is_checked=item.checkState()==Qt.CheckState.Checked;
             # Ensure the row index is valid
             if 0<=row_index<len(self.app_cfg['siem']['queries']):
                 current_enabled_state=self.app_cfg['siem']['queries'][row_index].get('enabled',True);
                 # Update config only if the state actually changed
                 if current_enabled_state != is_checked:
                     self.app_cfg['siem']['queries'][row_index]['enabled']=is_checked;
                     query_name = self.app_cfg['siem']['queries'][row_index].get('name',f'query at index {row_index}')
                     logger.debug(f"SIEM Query '{query_name}' enabled state changed to: {is_checked}");
                     self.mark_settings_dirty()
             else:
                 logger.error(f"SIEM query itemChanged signal received for invalid row index: {row_index}")

    def add_siem_query(self):
         dialog=QDialog(self); dialog.setWindowTitle("Add New SIEM Query");
         layout=QFormLayout(dialog);
         name_edit=QLineEdit(); name_edit.setPlaceholderText("Short, descriptive name for the query")
         query_edit=QTextEdit(); query_edit.setAcceptRichText(False); query_edit.setFixedHeight(100); query_edit.setPlaceholderText("Enter the full Splunk Search Processing Language (SPL) query here. Do not include the initial 'search ' command.")
         enabled_checkbox=QCheckBox("Enabled"); enabled_checkbox.setChecked(True);
         layout.addRow("Query Name*:",name_edit);
         layout.addRow("Search Query*:",query_edit);
         layout.addRow("", enabled_checkbox); # Checkbox below fields
         # Dialog buttons
         button_box=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);
         button_box.accepted.connect(dialog.accept);
         button_box.rejected.connect(dialog.reject);
         layout.addRow(button_box);

         if dialog.exec()==QDialog.DialogCode.Accepted:
              query_name=name_edit.text().strip();
              search_query=query_edit.toPlainText().strip()
              is_enabled=enabled_checkbox.isChecked()

              if query_name and search_query:
                  new_query_config={'name':query_name,'query':search_query,'enabled':is_enabled};
                  # Ensure SIEM structure exists
                  if 'siem' not in self.app_cfg: self.app_cfg['siem']={}
                  if 'queries' not in self.app_cfg['siem'] or not isinstance(self.app_cfg['siem']['queries'], list): self.app_cfg['siem']['queries']=[]
                  # Append the new query
                  self.app_cfg['siem']['queries'].append(new_query_config);
                  self.refresh_settings_ui(); # Update the table
                  self.mark_settings_dirty();
                  logger.info(f"Added new SIEM query: '{query_name}'")
                  self.notifications.show_message(f"SIEM Query '{query_name}' added. Apply & Save.",level="info")
              else:
                  QMessageBox.warning(self,"Input Error","Query Name and Search Query cannot be empty.")

    def edit_siem_query(self):
        selection_model=self.siem_query_table.selectionModel();
        selected_rows=selection_model.selectedRows();
        if not selected_rows:
             QMessageBox.information(self,"Edit SIEM Query","Please select a query from the table to edit."); return

        row_index=selected_rows[0].row(); # Get index of the first selected row
        # Validate row index
        if not (0<=row_index<len(self.app_cfg['siem']['queries'])):
             logger.error(f"Invalid row index {row_index} selected for editing SIEM query.")
             return;
        current_query_config=self.app_cfg['siem']['queries'][row_index]

        # Create and populate the edit dialog
        dialog=QDialog(self); dialog.setWindowTitle("Edit SIEM Query");
        layout=QFormLayout(dialog);
        name_edit=QLineEdit(current_query_config.get('name',''));
        query_edit=QTextEdit(current_query_config.get('query',''));
        query_edit.setAcceptRichText(False); query_edit.setFixedHeight(100);
        enabled_checkbox=QCheckBox("Enabled");
        enabled_checkbox.setChecked(current_query_config.get('enabled',True));
        layout.addRow("Query Name*:",name_edit);
        layout.addRow("Search Query*:",query_edit);
        layout.addRow("", enabled_checkbox);
        button_box=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);
        button_box.accepted.connect(dialog.accept);
        button_box.rejected.connect(dialog.reject);
        layout.addRow(button_box);

        if dialog.exec()==QDialog.DialogCode.Accepted:
            updated_name=name_edit.text().strip();
            updated_query=query_edit.toPlainText().strip()
            updated_enabled=enabled_checkbox.isChecked()

            if updated_name and updated_query:
                 updated_query_config={'name':updated_name,'query':updated_query,'enabled':updated_enabled};
                 # Update the config list at the specific index
                 self.app_cfg['siem']['queries'][row_index]=updated_query_config;
                 self.refresh_settings_ui(); # Update the table
                 self.mark_settings_dirty();
                 logger.info(f"Updated SIEM query: '{updated_name}'")
                 self.notifications.show_message(f"SIEM Query '{updated_name}' updated. Apply & Save.",level="info")
            else:
                 QMessageBox.warning(self,"Input Error","Query Name and Search Query cannot be empty.")

    def remove_siem_query(self):
         selection_model=self.siem_query_table.selectionModel();
         selected_rows=selection_model.selectedRows();
         if not selected_rows:
             QMessageBox.information(self,"Remove SIEM Query","Please select a query from the table to remove."); return

         row_index=selected_rows[0].row();
         if not (0<=row_index<len(self.app_cfg['siem']['queries'])):
              logger.error(f"Invalid row index {row_index} selected for removing SIEM query.")
              return;

         query_name_to_remove=self.app_cfg['siem']['queries'][row_index].get('name',f'query at index {row_index}')
         reply = QMessageBox.question(self,"Confirm Removal",f"Are you sure you want to remove the SIEM query '{query_name_to_remove}'?",
                                      QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)

         if reply==QMessageBox.StandardButton.Yes:
             del self.app_cfg['siem']['queries'][row_index]; # Remove from the list
             self.refresh_settings_ui(); # Update table
             self.mark_settings_dirty();
             logger.info(f"Removed SIEM query: '{query_name_to_remove}'")
             self.notifications.show_message(f"SIEM Query '{query_name_to_remove}' removed. Apply & Save.",level="info")

    def add_soar_playbook(self):
         dialog=QDialog(self); dialog.setWindowTitle("Add New SOAR Playbook");
         layout=QFormLayout(dialog);
         name_edit=QLineEdit(); name_edit.setPlaceholderText("User-friendly name for this playbook")
         id_edit=QLineEdit(); id_edit.setPlaceholderText("The ID or unique name the SOAR platform uses")
         context_edit=QLineEdit(); context_edit.setPlaceholderText("e.g., src_ip, file_hash, user_name")
         context_edit.setToolTip("Comma-separated list of SIEM field names required as input for this playbook.")
         layout.addRow("Playbook Name*:",name_edit);
         layout.addRow("Playbook ID*:",id_edit);
         layout.addRow("Required Context Fields:",context_edit);
         button_box=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);
         button_box.accepted.connect(dialog.accept);
         button_box.rejected.connect(dialog.reject);
         layout.addRow(button_box);

         if dialog.exec()==QDialog.DialogCode.Accepted:
              playbook_name=name_edit.text().strip();
              playbook_id=id_edit.text().strip();
              context_fields_str=context_edit.text().strip()
              context_fields_list = [f.strip() for f in context_fields_str.split(',') if f.strip()] # Split and clean

              if playbook_name and playbook_id:
                  new_playbook_config={'name':playbook_name,'id':playbook_id,'context_fields':context_fields_list};
                  # Ensure SOAR structure exists
                  if 'soar' not in self.app_cfg: self.app_cfg['soar']={}
                  if 'playbooks' not in self.app_cfg['soar'] or not isinstance(self.app_cfg['soar']['playbooks'], list): self.app_cfg['soar']['playbooks']=[]
                  # Append new playbook
                  self.app_cfg['soar']['playbooks'].append(new_playbook_config);
                  self.refresh_settings_ui(); # Update table
                  self.mark_settings_dirty();
                  logger.info(f"Added new SOAR playbook: '{playbook_name}'")
                  self.notifications.show_message(f"SOAR Playbook '{playbook_name}' added. Apply & Save.",level="info")
              else:
                  QMessageBox.warning(self,"Input Error","Playbook Name and Playbook ID cannot be empty.")

    def edit_soar_playbook(self):
         selection_model=self.soar_playbook_table.selectionModel();
         selected_rows=selection_model.selectedRows();
         if not selected_rows:
             QMessageBox.information(self,"Edit SOAR Playbook","Please select a playbook from the table to edit."); return

         row_index=selected_rows[0].row();
         if not (0<=row_index<len(self.app_cfg['soar']['playbooks'])):
              logger.error(f"Invalid row index {row_index} selected for editing SOAR playbook.")
              return;
         current_playbook_config=self.app_cfg['soar']['playbooks'][row_index]

         # Create and populate edit dialog
         dialog=QDialog(self); dialog.setWindowTitle("Edit SOAR Playbook");
         layout=QFormLayout(dialog);
         name_edit=QLineEdit(current_playbook_config.get('name',''));
         id_edit=QLineEdit(current_playbook_config.get('id',''));
         context_edit=QLineEdit(", ".join(current_playbook_config.get('context_fields',[]))); # Join list back to string
         context_edit.setToolTip("Comma-separated list of SIEM field names required as input for this playbook.")
         layout.addRow("Playbook Name*:",name_edit);
         layout.addRow("Playbook ID*:",id_edit);
         layout.addRow("Required Context Fields:",context_edit);
         button_box=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);
         button_box.accepted.connect(dialog.accept);
         button_box.rejected.connect(dialog.reject);
         layout.addRow(button_box);

         if dialog.exec()==QDialog.DialogCode.Accepted:
            updated_name=name_edit.text().strip();
            updated_id=id_edit.text().strip();
            updated_context_str=context_edit.text().strip()
            updated_context_list = [f.strip() for f in updated_context_str.split(',') if f.strip()] # Split and clean

            if updated_name and updated_id:
                 updated_playbook_config={'name':updated_name,'id':updated_id,'context_fields':updated_context_list};
                 # Update config list at the index
                 self.app_cfg['soar']['playbooks'][row_index]=updated_playbook_config;
                 self.refresh_settings_ui(); # Update table
                 self.mark_settings_dirty();
                 logger.info(f"Updated SOAR playbook: '{updated_name}'")
                 self.notifications.show_message(f"SOAR Playbook '{updated_name}' updated. Apply & Save.",level="info")
            else:
                 QMessageBox.warning(self,"Input Error","Playbook Name and Playbook ID cannot be empty.")

    def remove_soar_playbook(self):
         selection_model=self.soar_playbook_table.selectionModel();
         selected_rows=selection_model.selectedRows();
         if not selected_rows:
             QMessageBox.information(self,"Remove SOAR Playbook","Please select a playbook from the table to remove."); return

         row_index=selected_rows[0].row();
         if not (0<=row_index<len(self.app_cfg['soar']['playbooks'])):
              logger.error(f"Invalid row index {row_index} selected for removing SOAR playbook.")
              return;

         playbook_name_to_remove=self.app_cfg['soar']['playbooks'][row_index].get('name',f'playbook at index {row_index}')
         reply = QMessageBox.question(self,"Confirm Removal",f"Are you sure you want to remove the SOAR playbook '{playbook_name_to_remove}'?",
                                      QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)

         if reply==QMessageBox.StandardButton.Yes:
             del self.app_cfg['soar']['playbooks'][row_index]; # Remove from list
             self.refresh_settings_ui(); # Update table
             self.mark_settings_dirty();
             logger.info(f"Removed SOAR playbook: '{playbook_name_to_remove}'")
             self.notifications.show_message(f"SOAR Playbook '{playbook_name_to_remove}' removed. Apply & Save.",level="info")

    def apply_and_save_settings(self):
        logger.info("Applying and saving all settings...");
        # --- Read UI values back into the config dictionary ---
        # SIEM Connection Settings
        if 'siem' not in self.app_cfg: self.app_cfg['siem']={}
        siem_cfg = self.app_cfg['siem']
        siem_cfg['api_url']=self.siem_url_input.text().strip();
        siem_cfg['token']=self.siem_token_input.text(); # Assume token might be empty intentionally
        siem_cfg['auth_header']=self.siem_auth_combo.currentText();
        siem_cfg['verify_ssl']=self.siem_verify_ssl_check.isChecked()
        try: # Validate refresh interval
            interval = int(self.siem_refresh_input.text().strip() or '0')
            siem_cfg['refresh_interval_min']=max(0,interval)
        except ValueError:
            siem_cfg['refresh_interval_min']=15; # Default on error
            self.siem_refresh_input.setText("15"); # Correct UI
            logger.warning("Invalid SIEM refresh interval entered, defaulting to 15 minutes.")
        # Note: SIEM Queries are updated directly via itemChanged signal and add/edit/remove actions

        # SOAR Connection Settings
        if 'soar' not in self.app_cfg: self.app_cfg['soar']={}
        soar_cfg = self.app_cfg['soar']
        soar_cfg['enabled']=self.soar_enabled_check.isChecked();
        soar_cfg['api_url']=self.soar_api_url_input.text().strip();
        soar_cfg['api_key']=self.soar_api_key_input.text();
        soar_cfg['auth_header_name']=(self.soar_auth_header_name_input.text().strip() or 'Authorization'); # Default if empty
        soar_cfg['auth_header_prefix']=self.soar_auth_header_prefix_input.text(); # Allow empty prefix
        soar_cfg['verify_ssl']=self.soar_verify_ssl_check.isChecked()
        # Note: SOAR Playbooks are updated directly via add/edit/remove actions

        # Note: Camera configs are updated directly via add/edit/remove actions
        # Note: Map image path is updated directly via select_and_load_map_image

        # --- Validate the potentially modified config ---
        # (Validation also handles map positions based on current markers)
        self._validate_and_correct_config()

        # --- Save the configuration ---
        if self.save_config(): # save_config now also sets dirty flag to False
             self.notifications.show_message("Applying changes...",level="info",duration=1500);
             QApplication.processEvents() # Show message before potentially blocking operations

             # --- Re-initialize system components with the new config ---
             self.stop_all_cameras(); # Stop old processes
             self.init_system();      # Start new processes with updated config
             self.update_siem_timer_interval(); # Reset SIEM timer based on new interval
             self.recreate_monitor_tab(); # Rebuild monitor UI reflecting new cameras/status
             self.load_map_image() # Ensure map reflects current config (path & markers)

             self.notifications.show_message("Settings Applied & Saved Successfully!",level="success",duration=3500);
             self.mark_settings_dirty(False) # Explicitly ensure dirty is false after successful apply/save
        else:
             QMessageBox.critical(self,"Save Error","Failed to save the configuration file. Changes have not been fully applied. Please check file permissions and logs.")


    # ==================== Theme & Misc Methods ====================
    def apply_dark_theme(self):
        logger.debug("Applying dark theme...");
        dark_palette=QPalette();
        # Define colors (adjust as desired)
        COLOR_WINDOW_BG = QColor(53,53,53); COLOR_WINDOW_FG = QColor(230,230,230);
        COLOR_BASE_BG = QColor(35,35,35);   COLOR_ALT_BASE_BG = QColor(45,45,45);
        COLOR_TOOLTIP_BG = QColor(25,25,25); COLOR_TOOLTIP_FG = QColor(230,230,230);
        COLOR_TEXT_FG = QColor(220,220,220);
        COLOR_BUTTON_BG = QColor(66,66,66); COLOR_BUTTON_FG = QColor(230,230,230);
        COLOR_BUTTON_DISABLED_FG = QColor(127,127,127);
        COLOR_BRIGHT_TEXT = QColor(255,80,80); # For emphasis?
        COLOR_HIGHLIGHT_BG = QColor(42,130,218); COLOR_HIGHLIGHT_FG = QColor(255,255,255);
        COLOR_HIGHLIGHT_DISABLED_BG = QColor(80,80,80);
        COLOR_LINK = QColor(80,160,240); COLOR_LINK_VISITED = QColor(160,100,220);
        COLOR_BORDER = QColor(80,80,80);

        # Apply colors to palette roles
        dark_palette.setColor(QPalette.ColorRole.Window,COLOR_WINDOW_BG);
        dark_palette.setColor(QPalette.ColorRole.WindowText,COLOR_WINDOW_FG);
        dark_palette.setColor(QPalette.ColorRole.Base,COLOR_BASE_BG);
        dark_palette.setColor(QPalette.ColorRole.AlternateBase,COLOR_ALT_BASE_BG);
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase,COLOR_TOOLTIP_BG);
        dark_palette.setColor(QPalette.ColorRole.ToolTipText,COLOR_TOOLTIP_FG);
        dark_palette.setColor(QPalette.ColorRole.Text,COLOR_TEXT_FG);
        dark_palette.setColor(QPalette.ColorRole.Button,COLOR_BUTTON_BG);
        dark_palette.setColor(QPalette.ColorRole.ButtonText,COLOR_BUTTON_FG);
        dark_palette.setColor(QPalette.ColorGroup.Disabled,QPalette.ColorRole.ButtonText,COLOR_BUTTON_DISABLED_FG);
        dark_palette.setColor(QPalette.ColorRole.BrightText,COLOR_BRIGHT_TEXT);
        dark_palette.setColor(QPalette.ColorRole.Highlight,COLOR_HIGHLIGHT_BG);
        dark_palette.setColor(QPalette.ColorRole.HighlightedText,COLOR_HIGHLIGHT_FG);
        dark_palette.setColor(QPalette.ColorGroup.Disabled,QPalette.ColorRole.Highlight,COLOR_HIGHLIGHT_DISABLED_BG);
        dark_palette.setColor(QPalette.ColorRole.Link,COLOR_LINK);
        dark_palette.setColor(QPalette.ColorRole.LinkVisited,COLOR_LINK_VISITED)
        # Set disabled text color explicitly
        dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, COLOR_BUTTON_DISABLED_FG)

        # Stylesheet for finer control (adjust border colors, gradients, etc.)
        stylesheet=f"""
        QWidget {{
            font-size: 9pt;
            border-color: {COLOR_BORDER.name()};
        }}
        QMainWindow, QDialog {{
            background-color: {COLOR_WINDOW_BG.name()};
        }}
        QToolTip {{
            color: {COLOR_TOOLTIP_FG.name()};
            background-color: {COLOR_TOOLTIP_BG.name()};
            border: 1px solid #3b3b3b; /* Slightly darker border */
            padding: 5px;
            border-radius: 3px;
            opacity: 230; /* Semi-transparent */
        }}
        QGroupBox {{
            font-weight: bold;
            color: #ddd; /* Slightly brighter group title */
            border: 1px solid {COLOR_BORDER.name()};
            border-radius: 6px;
            margin-top: 0.6em; /* Space for title */
            padding: 0.8em 0.5em 0.5em 0.5em; /* Padding inside box */
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px; /* Padding around title text */
            left: 10px; /* Indent title slightly */
            color: #ccc; /* Lighter title text */
        }}
        QTabWidget::pane {{
            border: 1px solid {COLOR_BORDER.darker(110).name()}; /* Slightly darker border for pane */
            border-radius: 3px;
            margin-top: -1px; /* Overlap with tab bar */
            background-color: {COLOR_BASE_BG.name()};
        }}
        QTabBar::tab {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #666, stop:1 #555); /* Gradient for tabs */
            border: 1px solid {COLOR_BORDER.darker(110).name()};
            border-bottom: none; /* No bottom border for non-selected */
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
            min-width: 8ex;
            padding: 6px 10px;
            margin-right: 2px;
            color: #ccc; /* Non-selected tab text color */
        }}
        QTabBar::tab:selected {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a4a4a, stop:1 {COLOR_BASE_BG.name()}); /* Different gradient for selected */
            border-color: {COLOR_BORDER.darker(110).name()};
            color: #fff; /* Selected tab text color */
            font-weight: bold;
        }}
        QTabBar::tab:!selected {{
            margin-top: 2px; /* Push non-selected tabs down slightly */
            background: #555;
        }}
        QTabBar::tab:!selected:hover {{
            background: #777; /* Hover effect */
            color: #fff;
        }}
        QPushButton {{
            color: {COLOR_BUTTON_FG.name()};
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #666, stop:1 #5a5a5a);
            border: 1px solid {COLOR_BORDER.name()};
            border-radius: 4px;
            padding: 6px 12px;
            min-width: 60px;
        }}
        QPushButton:hover {{
            background-color: #777;
            border-color: {COLOR_BORDER.lighter(110).name()};
        }}
        QPushButton:pressed {{
            background-color: #505050; /* Darker when pressed */
        }}
        QPushButton:checked {{ /* For toggle buttons like Map Edit */
            background-color: {COLOR_HIGHLIGHT_BG.name()};
            border-color: {COLOR_HIGHLIGHT_BG.darker(120).name()};
            color: {COLOR_HIGHLIGHT_FG.name()};
        }}
        QPushButton:disabled {{
            color: {COLOR_BUTTON_DISABLED_FG.name()};
            background-color: #444;
            border-color: #555;
        }}
        QLineEdit, QComboBox, QAbstractSpinBox, QTextEdit {{
            color: {COLOR_TEXT_FG.name()};
            background-color: {COLOR_ALT_BASE_BG.name()}; /* Slightly lighter background for inputs */
            border: 1px solid {COLOR_BORDER.name()};
            border-radius: 4px;
            padding: 4px 6px;
        }}
        QTextEdit {{
            padding: 5px; /* More padding for text edits */
        }}
        QComboBox::drop-down {{
            border: none; /* Remove border around dropdown arrow */
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 18px;
            /* Consider adding an ::arrow subcontrol if needed */
        }}
        QComboBox QAbstractItemView {{ /* Style the dropdown list */
            background-color: {COLOR_ALT_BASE_BG.name()};
            border: 1px solid {COLOR_BORDER.name()};
            selection-background-color: {COLOR_HIGHLIGHT_BG.name()};
            selection-color: {COLOR_HIGHLIGHT_FG.name()};
            color: {COLOR_TEXT_FG.name()};
            outline: 0px; /* Remove focus outline */
        }}
        QListWidget, QTableWidget {{
            color: {COLOR_TEXT_FG.name()};
            background-color: {COLOR_BASE_BG.name()};
            border: 1px solid {COLOR_BORDER.name()};
            border-radius: 4px;
            padding: 2px;
            alternate-background-color: {COLOR_ALT_BASE_BG.name()};
            gridline-color: {COLOR_BORDER.name()};
        }}
        QListWidget::item, QTableWidget::item {{
            padding: 4px 2px;
            border: none; /* No border around individual items */
        }}
        QListWidget::item:selected, QTableWidget::item:selected {{
            background-color: {COLOR_HIGHLIGHT_BG.name()};
            color: {COLOR_HIGHLIGHT_FG.name()};
        }}
        QListWidget::item:selected:!active, QTableWidget::item:selected:!active {{
             background-color: {COLOR_HIGHLIGHT_BG.darker(120).name()}; /* Selection color when widget not focused */
        }}
        QTableWidget QHeaderView::section {{
            background-color: {COLOR_BUTTON_BG.name()}; /* Header background */
            color: {COLOR_BUTTON_FG.name()};
            padding: 4px;
            border: 1px solid {COLOR_BORDER.name()};
            font-weight: bold;
        }}
        QCheckBox {{
            spacing: 8px; /* Space between indicator and text */
        }}
        QCheckBox::indicator {{
            width: 16px; height: 16px;
            border: 1px solid {COLOR_BORDER.name()};
            border-radius: 4px;
            background-color: {COLOR_ALT_BASE_BG.name()};
        }}
        QCheckBox::indicator:checked {{
            background-color: {COLOR_HIGHLIGHT_BG.name()};
            border-color: {COLOR_HIGHLIGHT_BG.darker(120).name()};
            /* image: url(path/to/checkmark.png); Optional checkmark image */
        }}
        QCheckBox::indicator:disabled {{
            background-color: #444;
            border-color: #555;
        }}
        QToolBar {{
            background-color: {COLOR_WINDOW_BG.darker(110).name()}; /* Darker toolbar */
            border: none;
            padding: 3px;
            spacing: 4px;
        }}
        QToolButton {{
            background-color: transparent;
            border: none;
            padding: 4px;
            border-radius: 4px;
            color: {COLOR_BUTTON_FG.name()};
        }}
        QToolButton:hover {{
            background-color: {COLOR_BUTTON_BG.lighter(120).name()};
        }}
        QToolButton:pressed {{
            background-color: {COLOR_BUTTON_BG.name()};
        }}
        QToolButton:checked {{ /* For checkable toolbar actions like Pan/Edit */
            background-color: {COLOR_HIGHLIGHT_BG.name()};
            border: 1px solid {COLOR_HIGHLIGHT_BG.darker(120).name()};
            color: {COLOR_HIGHLIGHT_FG.name()};
        }}
        QStatusBar {{
            color: #bbb; /* Status bar text color */
        }}
        QStatusBar::item {{
            border: none; /* No border around status bar items */
        }}
        QGraphicsView {{
            border: 1px solid {COLOR_BORDER.name()};
            border-radius: 3px;
            background-color: {COLOR_ALT_BASE_BG.darker(110).name()}; /* Dark background for map area */
        }}
        QScrollArea {{ /* Ensure scroll areas have no extra border */
            border: none;
        }}
        QScrollBar:vertical {{
            border: 1px solid {COLOR_BORDER.name()};
            background: {COLOR_BASE_BG.name()};
            width: 12px; margin: 0px;
        }}
        QScrollBar::handle:vertical {{
            background: {COLOR_BUTTON_BG.name()};
            min-height: 20px; border-radius: 5px;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px; background: none; /* Hide arrows */
        }}
        QScrollBar:horizontal {{
            border: 1px solid {COLOR_BORDER.name()};
            background: {COLOR_BASE_BG.name()};
            height: 12px; margin: 0px;
        }}
        QScrollBar::handle:horizontal {{
            background: {COLOR_BUTTON_BG.name()};
            min-width: 20px; border-radius: 5px;
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px; background: none; /* Hide arrows */
        }}
        QMenu {{
            background-color: {COLOR_ALT_BASE_BG.name()};
            border: 1px solid {COLOR_BORDER.darker(120).name()};
            color: {COLOR_TEXT_FG.name()};
            padding: 4px; /* Padding around menu content */
        }}
        QMenu::item {{
            padding: 5px 20px 5px 20px; /* Padding inside items */
        }}
        QMenu::item:selected {{
            background-color: {COLOR_HIGHLIGHT_BG.name()};
            color: {COLOR_HIGHLIGHT_FG.name()};
        }}
        QMenu::separator {{
            height: 1px;
            background: {COLOR_BORDER.name()};
            margin: 4px 0px; /* Spacing around separator */
        }}
        """
        app=QApplication.instance();
        if app:
            app.setPalette(dark_palette);
            app.setStyleSheet(stylesheet);
            logger.debug("Dark theme applied via palette and stylesheet.")

    def show_about(self):
        try:
            python_version=platform.python_version();
            pyqt_version=Qt.PYQT_VERSION_STR;
            opencv_version=cv2.__version__;
            # Use importlib.metadata for package versions (more robust)
            import importlib.metadata
            try: onvif_version=importlib.metadata.version('onvif_zeep');
            except importlib.metadata.PackageNotFoundError: onvif_version = "Not Found"
            try: requests_version=importlib.metadata.version('requests')
            except importlib.metadata.PackageNotFoundError: requests_version = "Not Found"

        except Exception as e:
            logger.error(f"Error getting dependency versions for About dialog: {e}")
            python_version=pyqt_version=opencv_version=onvif_version=requests_version="N/A"

        app_version="1.4.4" # Define app version here
        about_text=f"""
        <h2>Security Monitor Pro</h2>
        <p>Version: {app_version}</p>
        <p>A dashboard for monitoring security cameras and SIEM/SOAR integration.</p>
        <hr>
        <p><b>Core Dependencies:</b></p>
        <ul>
            <li>Python: {python_version}</li>
            <li>PyQt6: {pyqt_version}</li>
            <li>OpenCV: {opencv_version}</li>
            <li>ONVIF-Zeep: {onvif_version}</li>
            <li>Requests: {requests_version}</li>
            <li>Operating System: {platform.system()} ({platform.release()})</li>
        </ul>
        <hr>
        <p><b>Key Features:</b></p>
        <ul>
            <li>RTSP & ONVIF Camera Support (including PTZ)</li>
            <li>SIEM Event Fetching (Splunk API Example)</li>
            <li>SOAR Playbook Triggering (Splunk SOAR Example)</li>
            <li>Basic Motion Detection</li>
            <li>Interactive Map View for Camera Layout</li>
            <li>Dynamic Camera Configuration</li>
            <li>Snapshot Saving</li>
            <li>Automatic Dependency Installation</li>
            <li>Dark Theme</li>
        </ul>
        """
        QMessageBox.about(self, f"About Security Monitor Pro v{app_version}", about_text)


    def resizeEvent(self, event:'QResizeEvent'):
        super().resizeEvent(event);
        # Keep notification centered horizontally and pinned near top
        if hasattr(self,'notifications') and self.notifications and self.notifications.isVisible():
            parent_width=self.central_widget.width();
            notification_width=self.notifications.width();
            notification_height=self.notifications.height();
            new_x=(parent_width - notification_width)//2;
            current_y=self.notifications.geometry().y();
            # Keep Y position unless it's off-screen (e.g., during initial show animation)
            new_y=max(20,current_y); # Ensure it's at least 20px from top
            self.notifications.setGeometry(new_x,new_y,notification_width,notification_height)

    def check_unsaved_changes(self, action_description:str="perform this action") -> bool:
         """Checks if config is dirty and prompts user to save, discard, or cancel."""
         if not self._dirty: return True # No changes, safe to proceed

         reply = QMessageBox.question(self,"Unsaved Changes",
                                     f"You have unsaved configuration changes. Save before {action_description}?",
                                      QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                                      QMessageBox.StandardButton.Cancel) # Default to Cancel

         if reply == QMessageBox.StandardButton.Save:
             return self.save_config() # Proceed only if save is successful
         elif reply == QMessageBox.StandardButton.Discard:
             logger.info("Discarding unsaved configuration changes.")
             self.mark_settings_dirty(False); # Reset dirty flag
             return True # Proceed without saving
         else: # Cancelled
             logger.info(f"Action '{action_description}' cancelled due to unsaved changes.")
             return False # Do not proceed


    def closeEvent(self, event:'QCloseEvent'):
        logger.info("Application close requested...");
        # Check for unsaved changes before closing
        if not self.check_unsaved_changes("exiting the application"):
            event.ignore(); # Abort the close event
            logger.info("Application close cancelled by user or save failure.");
            return

        # Proceed with shutdown
        if hasattr(self,'status_bar'): self.status_bar.showMessage("Shutting down application...");
        QApplication.processEvents() # Update UI

        if hasattr(self,'siem_timer') and self.siem_timer.isActive():
             self.siem_timer.stop(); logger.debug("SIEM refresh timer stopped.")

        self.stop_all_cameras(); # Gracefully stop camera threads

        logger.info("Shutdown sequence complete. Exiting application.");
        event.accept() # Allow the window to close


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # Enable High DPI scaling on Windows if appropriate Qt version
    if platform.system()=="Windows":
        # Check if Qt supports High DPI Scaling attribute (Qt 5.6+)
        if hasattr(Qt, 'AA_EnableHighDpiScaling'):
            QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
            logger.info("Enabled High DPI Scaling.")
        if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
             QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
             logger.info("Enabled High DPI Pixmaps.")

    QApplication.setApplicationName("SecurityMonitorPro")
    QApplication.setOrganizationName("UserProject") # Optional: For settings persistence

    app = QApplication(sys.argv)
    # Use Fusion style for a consistent look across platforms
    app.setStyle('Fusion')

    window = None # Define window variable outside try block
    try:
        logger.info("Creating main application window...")
        window = SecurityMonitorApp() # Instantiates the main class
        window.show()                # Make the window visible
        logger.info("Starting Qt application event loop.")
        exit_code = app.exec()       # Run the application
        logger.info(f"Application event loop finished. Exit code: {exit_code}.")
        sys.exit(exit_code)          # Exit Python script with the application's exit code

    except Exception as e:
         # Log the critical error
         logger.critical(f"A fatal error occurred during application execution: {e}", exc_info=True)
         # Attempt to show a critical error message box
         try:
             # Ensure a QApplication instance exists for the message box
             if not QApplication.instance(): temp_app=QApplication([])
             QMessageBox.critical(None,"Fatal Application Error",
                                  f"A critical error occurred:\n{e}\n\nThe application must now exit. Please check the log files for details.")
         except Exception as msg_e:
             # Fallback to printing to stderr if GUI message box fails
             print(f"FATAL APPLICATION ERROR: {e}\n(Additionally, failed to show GUI error message: {msg_e})", file=sys.stderr)
         sys.exit(1) # Exit with a non-zero code indicating an error
