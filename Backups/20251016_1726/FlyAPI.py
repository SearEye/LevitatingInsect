
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlyAPI.py (compact stable build)
- Camera device refresh button
- Dropdown selectors per camera
- Toggleable, pop-out previews (off by default)
- High-FPS capture threads (only run when preview ON or recording)
- Recording at source resolution
- Simple "looming / falling object" stimulus (growing dot) with monitor selection + fullscreen
- Robust PySpin handling: single System instance for entire app lifetime; no mid-run ReleaseInstance
- Conservative OpenCV index probing (0..3) to avoid MSMF/DSHOW spam and crashes
"""
import os, sys, time, atexit, threading, queue, pathlib, math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

APP_VERSION = "1.33.0"

# ------------------------------ Imports ---------------------------------------
try:
    import cv2
    HAS_CV2 = True
except Exception as e:
    HAS_CV2 = False
    cv2 = None
    print(f"[BOOT] OpenCV import failed: {e}")

from PyQt5 import QtCore, QtGui, QtWidgets

# PsychoPy optional; app still works without it
try:
    from psychopy import visual, core, monitors, event
    HAS_PSYCHOPY = True
except Exception:
    HAS_PSYCHOPY = False

# PySpin optional
try:
    import PySpin
    HAS_PYSPIN = True
except Exception:
    HAS_PYSPIN = False
    PySpin = None

# ------------------------------ Logging helper --------------------------------
def banner():
    print("=== FlyPy Startup ===")
    print(f"Version: {APP_VERSION}")
    print(f"OpenCV: {'OK' if HAS_CV2 else 'not available'}")
    print(f"PsychoPy: {'OK' if HAS_PSYCHOPY else 'not available (OpenCV stimulus fallback)'}")
    print(f"PySpin: {'OK' if HAS_PYSPIN else 'not available'}")
    print("======================")

# ------------------------------ Backends --------------------------------------
CV_BACKENDS = []
if HAS_CV2:
    # Prefer DirectShow on Windows; fallback to MSMF. Limit probing noise.
    if hasattr(cv2, "CAP_DSHOW"):
        CV_BACKENDS.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):
        CV_BACKENDS.append(cv2.CAP_MSMF)

_SPIN_SYSTEM = None
def _spin_get_system():
    global _SPIN_SYSTEM
    if not HAS_PYSPIN:
        return None
    if _SPIN_SYSTEM is None:
        _SPIN_SYSTEM = PySpin.System.GetInstance()
    return _SPIN_SYSTEM

def _spin_release_system():
    global _SPIN_SYSTEM
    if _SPIN_SYSTEM is not None:
        try:
            # Ensure ALL refs are gone first
            # No camera or nodemap objects should be live here.
            _SPIN_SYSTEM.ReleaseInstance()
            print("[PySpin] System.ReleaseInstance() OK (app exit)")
        except Exception as e:
            print(f"[PySpin] ReleaseInstance warning at exit: {e}")
        _SPIN_SYSTEM = None
atexit.register(_spin_release_system)

# ----------------------------- Data classes -----------------------------------
@dataclass
class DeviceInfo:
    backend: str           # "PySpin" or "OpenCV"
    ident: str             # serial or index as string
    display: str           # nice name

@dataclass
class OpenCVCam:
    index: int
    backend: int
    cap: Optional['cv2.VideoCapture'] = field(default=None, init=False)
    lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    size: Tuple[int,int] = field(default=(0,0), init=False)

    @staticmethod
    def enumerate(max_indices: int = 4) -> List['OpenCVCam']:
        cams = []
        for idx in range(max_indices):
            backend_found = None
            for be in CV_BACKENDS:
                cap = cv2.VideoCapture(idx, be)
                if cap is not None and cap.isOpened():
                    backend_found = be
                    cap.release()
                    break
            if backend_found is not None:
                cams.append(OpenCVCam(idx, backend_found))
        return cams

    def open(self) -> bool:
        with self.lock:
            if self.cap and self.cap.isOpened():
                return True
            self.cap = cv2.VideoCapture(self.index, self.backend)
            if not self.cap or not self.cap.isOpened():
                print(f"[cam{self.index}] OpenCV open failed (backend={self.backend})")
                self.cap = None
                return False
            # Query size
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if not w or not h:
                # Default query frame to establish size
                ok, frame = self.cap.read()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
            self.size = (w, h)
            print(f"[cam{self.index}] OpenCV open: index {self.index} ({w}x{h})")
            return True

    def read(self):
        with self.lock:
            if not self.cap or not self.cap.isOpened():
                return False, None
            return self.cap.read()

    def close(self):
        with self.lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = None

@dataclass
class PySpinCam:
    serial: str
    lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    cam: Optional['PySpin.CameraPtr'] = field(default=None, init=False)
    size: Tuple[int,int] = field(default=(0,0), init=False)
    _acquiring: bool = field(default=False, init=False)

    @staticmethod
    def enumerate() -> List['PySpinCam']:
        sysm = _spin_get_system()
        if sysm is None:
            return []
        lst = sysm.GetCameras()
        res: List[PySpinCam] = []
        try:
            n = lst.GetSize()
            for i in range(n):
                c = lst.GetByIndex(i)
                try:
                    nodemap = c.GetTLDeviceNodeMap()
                    node = PySpin.CStringPtr(nodemap.GetNode("DeviceSerialNumber"))
                    serial = str(node.GetValue()) if PySpin.IsAvailable(node) and PySpin.IsReadable(node) else f"idx:{i}"
                except Exception:
                    serial = f"idx:{i}"
                res.append(PySpinCam(serial))
        finally:
            try:
                lst.Clear()
            except Exception:
                pass
        return res

    def open(self) -> bool:
        if not HAS_PYSPIN:
            return False
        sysm = _spin_get_system()
        if sysm is None:
            return False
        with self.lock:
            if self.cam is not None:
                return True
            lst = sysm.GetCameras()
            target = None
            try:
                n = lst.GetSize()
                for i in range(n):
                    c = lst.GetByIndex(i)
                    nodemap = c.GetTLDeviceNodeMap()
                    node = PySpin.CStringPtr(nodemap.GetNode("DeviceSerialNumber"))
                    serial = str(node.GetValue()) if PySpin.IsAvailable(node) and PySpin.IsReadable(node) else f"idx:{i}"
                    if serial == self.serial or (self.serial.startswith("idx:") and self.serial == f"idx:{i}"):
                        target = c
                        break
                if target is None:
                    print(f"[PySpin] Serial {self.serial} not found; using first camera if available.")
                    if n > 0:
                        target = lst.GetByIndex(0)
                if target is None:
                    return False
                self.cam = target
                self.cam.Init()
                # Leave default ROI; only get size for writer
                s_node = self.cam.GetNodeMap().GetNode("Width")
                w = int(s_node.GetValue()) if PySpin.IsAvailable(s_node) and PySpin.IsReadable(s_node) else 0
                s_node = self.cam.GetNodeMap().GetNode("Height")
                h = int(s_node.GetValue()) if PySpin.IsAvailable(s_node) and PySpin.IsReadable(s_node) else 0
                if not w or not h:
                    w = 640; h = 480
                self.size = (w, h)
                # Prepare Continuous acquisition
                node_acq = self.cam.GetNodeMap().GetNode("AcquisitionMode")
                if PySpin.IsAvailable(node_acq) and PySpin.IsWritable(node_acq):
                    node = PySpin.CEnumerationPtr(node_acq)
                    entry = node.GetEntryByName("Continuous")
                    node.SetIntValue(entry.GetValue())
                print(f"[PySpin] open: serial={self.serial} size={self.size}")
                return True
            finally:
                try:
                    lst.Clear()
                except Exception:
                    pass

    def start_acq(self):
        with self.lock:
            if self.cam and not self._acquiring:
                self.cam.BeginAcquisition()
                self._acquiring = True

    def get_frame(self, timeout_ms=50):
        with self.lock:
            if not self.cam or not self._acquiring:
                return False, None
            try:
                img = self.cam.GetNextImage(timeout_ms)
                if img.IsIncomplete():
                    img.Release()
                    return False, None
                arr = img.GetNDArray()  # mono or color depending on camera
                img.Release()
                return True, arr
            except PySpin.SpinnakerException:
                return False, None

    def stop_acq(self):
        with self.lock:
            if self.cam and self._acquiring:
                try:
                    self.cam.EndAcquisition()
                except Exception:
                    pass
                self._acquiring = False

    def close(self):
        with self.lock:
            try:
                self.stop_acq()
            except Exception:
                pass
            if self.cam:
                try:
                    self.cam.DeInit()
                except Exception:
                    pass
            self.cam = None

# ----------------------------- Worker threads ---------------------------------
class CameraWorker(QtCore.QThread):
    newFrame = QtCore.pyqtSignal(int, object)   # cam_index, numpy array (BGR)
    fpsUpdate = QtCore.pyqtSignal(int, float)   # cam_index, fps

    def __init__(self, cam_id: int, device: DeviceInfo, parent=None):
        super().__init__(parent)
        self.cam_id = cam_id
        self.device = device
        self.running = False
        self.recording = False
        self.writer = None
        self._last_ts = None
        self._frames = 0
        self._last_fps_emit = time.time()
        self._lock = threading.RLock()
        self._cv = None
        self._ps = None

    def run(self):
        self.running = True
        if self.device.backend == "OpenCV":
            idx = int(self.device.ident)
            be = CV_BACKENDS[0] if CV_BACKENDS else 0
            self._cv = OpenCVCam(idx, be)
            if not self._cv.open():
                print(f"[Worker{self.cam_id}] OpenCV open failed")
                self.running = False
        else:
            self._ps = PySpinCam(self.device.ident)
            if not self._ps.open():
                print(f"[Worker{self.cam_id}] PySpin open failed")
                self.running = False
            else:
                self._ps.start_acq()

        while self.running:
            ok, frame = (False, None)
            if self._cv:
                ok, frame = self._cv.read()
            elif self._ps:
                ok, frame = self._ps.get_frame(timeout_ms=30)
                # PySpin NDArray might be mono; ensure BGR for preview/writer
                if ok and frame is not None:
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame.shape[2] == 1:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            if not ok or frame is None:
                # small nap to avoid tight loop
                time.sleep(0.005)
                continue

            # FPS measurement
            now = time.time()
            if self._last_ts is None:
                self._last_ts = now
                self._frames = 0
            self._frames += 1
            if now - self._last_fps_emit >= 0.5:
                fps = self._frames / max(1e-6, now - self._last_ts)
                self.fpsUpdate.emit(self.cam_id, fps)
                self._last_fps_emit = now
                self._frames = 0
                self._last_ts = now

            # Emit frame for preview (receiver decides whether to paint)
            self.newFrame.emit(self.cam_id, frame)

            # Write to disk if recording
            with self._lock:
                if self.recording and self.writer is not None:
                    self.writer.write(frame)

        # Cleanup
        with self._lock:
            if self.writer is not None:
                try:
                    self.writer.release()
                except Exception:
                    pass
            self.writer = None
        if self._cv:
            self._cv.close()
            self._cv = None
        if self._ps:
            try:
                self._ps.close()
            except Exception:
                pass
            self._ps = None

    def start_recording(self, output_path: str, fps_hint: float = 120.0):
        with self._lock:
            # Writer expects color frames
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            # If camera opened, ask for its size
            w, h = (0,0)
            if self._cv and self._cv.size != (0,0):
                w, h = self._cv.size
            elif self._ps and self._ps.size != (0,0):
                w, h = self._ps.size
            # If unknown, we will set from first frame we see (lazy); but here set a default
            if not w or not h:
                w, h = (640, 480)
            self.writer = cv2.VideoWriter(output_path, fourcc, max(1.0, fps_hint), (w, h), True)
            if not self.writer.isOpened():
                print(f"[Worker{self.cam_id}] VideoWriter failed to open: {output_path}")
                self.writer = None
                return False
            self.recording = True
            print(f"[Worker{self.cam_id}] Recording → {output_path} at ~{fps_hint:.1f} FPS, size=({w}x{h})")
            return True

    def stop_recording(self):
        with self._lock:
            self.recording = False
            if self.writer is not None:
                try:
                    self.writer.release()
                except Exception:
                    pass
                self.writer = None
            print(f"[Worker{self.cam_id}] Recording stopped.")

# ----------------------------- Stimulus window --------------------------------
class StimulusWindow(QtWidgets.QDialog):
    def __init__(self, screen: QtGui.QScreen, fullscreen: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Looming Stimulus")
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self._t0 = time.time()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(16) # ~60Hz

        # Place on target screen
        if screen is not None:
            geo = screen.geometry()
            self.setGeometry(geo)  # fill screen area; toggle fullscreen below
            if fullscreen:
                self.showFullScreen()
            else:
                # center window 800x600 on that screen
                w, h = 800, 600
                x = geo.x() + (geo.width()-w)//2
                y = geo.y() + (geo.height()-h)//2
                self.setGeometry(x, y, w, h)

    def paintEvent(self, e: QtGui.QPaintEvent):
        qp = QtGui.QPainter(self)
        qp.fillRect(self.rect(), QtCore.Qt.black)
        elapsed = time.time() - self._t0
        # radius grows: r(t) = r0 + k * t^2 (falling object illusion)
        r = int(10 + 100 * elapsed * elapsed)
        cx, cy = self.width()//2, self.height()//2
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
        qp.setBrush(QtGui.QBrush(QtCore.Qt.white))
        qp.setPen(QtCore.Qt.NoPen)
        qp.drawEllipse(QtCore.QPoint(cx, cy), r, r)
        qp.end()

# ----------------------------- GUI --------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FlyPy - Camera & Stimulus")
        self.resize(1100, 700)

        self.devices: List[DeviceInfo] = []
        self.cam_sel = [None, None]   # device indices into self.devices
        self.workers: List[Optional[CameraWorker]] = [None, None]
        self.preview_on = [False, False]

        # Central widget
        cw = QtWidgets.QWidget(self)
        self.setCentralWidget(cw)

        # Top controls
        top = QtWidgets.QHBoxLayout()
        self.btn_refresh = QtWidgets.QPushButton("Refresh cameras")
        self.btn_refresh.clicked.connect(self.refresh_devices)
        top.addWidget(self.btn_refresh)

        top.addSpacing(20)
        top.addWidget(QtWidgets.QLabel("Cam0:"))
        self.cb_cam0 = QtWidgets.QComboBox()
        top.addWidget(self.cb_cam0)
        self.btn_open0 = QtWidgets.QPushButton("Open 0")
        self.btn_open0.clicked.connect(lambda: self.open_cam(0))
        top.addWidget(self.btn_open0)
        self.btn_prev0 = QtWidgets.QPushButton("Toggle Preview 0")
        self.btn_prev0.clicked.connect(lambda: self.toggle_preview(0))
        top.addWidget(self.btn_prev0)
        self.lbl_fps0 = QtWidgets.QLabel("FPS0: -")
        top.addWidget(self.lbl_fps0)

        top.addSpacing(20)
        top.addWidget(QtWidgets.QLabel("Cam1:"))
        self.cb_cam1 = QtWidgets.QComboBox()
        top.addWidget(self.cb_cam1)
        self.btn_open1 = QtWidgets.QPushButton("Open 1")
        self.btn_open1.clicked.connect(lambda: self.open_cam(1))
        top.addWidget(self.btn_open1)
        self.btn_prev1 = QtWidgets.QPushButton("Toggle Preview 1")
        self.btn_prev1.clicked.connect(lambda: self.toggle_preview(1))
        top.addWidget(self.btn_prev1)
        self.lbl_fps1 = QtWidgets.QLabel("FPS1: -")
        top.addWidget(self.lbl_fps1)

        top.addStretch(1)

        # Recording controls
        rec = QtWidgets.QHBoxLayout()
        rec.addWidget(QtWidgets.QLabel("Output folder:"))
        self.ed_out = QtWidgets.QLineEdit(str(pathlib.Path("FlyPy_Output").absolute()))
        rec.addWidget(self.ed_out)
        self.btn_browse = QtWidgets.QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._browse_out)
        rec.addWidget(self.btn_browse)
        rec.addWidget(QtWidgets.QLabel("Duration (s):"))
        self.ed_dur = QtWidgets.QDoubleSpinBox()
        self.ed_dur.setRange(0.1, 3600.0)
        self.ed_dur.setValue(2.0)
        rec.addWidget(self.ed_dur)
        self.btn_rec = QtWidgets.QPushButton("Record both")
        self.btn_rec.clicked.connect(self.record_both)
        rec.addWidget(self.btn_rec)
        rec.addStretch(1)

        # Stimulus controls
        stim = QtWidgets.QHBoxLayout()
        stim.addWidget(QtWidgets.QLabel("Stimulus screen:"))
        self.cb_screens = QtWidgets.QComboBox()
        stim.addWidget(self.cb_screens)
        self.chk_full = QtWidgets.QCheckBox("Fullscreen")
        self.chk_full.setChecked(True)
        stim.addWidget(self.chk_full)
        self.btn_stim = QtWidgets.QPushButton("Show Looming Dot")
        self.btn_stim.clicked.connect(self.show_stimulus)
        stim.addWidget(self.btn_stim)
        stim.addStretch(1)

        # Hidden previews (popout)
        prev = QtWidgets.QHBoxLayout()
        self.lbl0 = QtWidgets.QLabel("Preview 0 (hidden)"); self.lbl0.setFixedSize(400, 300); self.lbl0.setVisible(False)
        self.lbl1 = QtWidgets.QLabel("Preview 1 (hidden)"); self.lbl1.setFixedSize(400, 300); self.lbl1.setVisible(False)
        prev.addWidget(self.lbl0); prev.addWidget(self.lbl1); prev.addStretch(1)

        # Layout
        v = QtWidgets.QVBoxLayout(cw)
        v.addLayout(top)
        v.addSpacing(8)
        v.addLayout(rec)
        v.addSpacing(8)
        v.addLayout(stim)
        v.addSpacing(8)
        v.addLayout(prev)

        # screen list
        self._refresh_screens()

        # signals for worker FPS and frames
        # Workers will be created after device selection/open
        # Fill device lists initially
        QtCore.QTimer.singleShot(0, self.refresh_devices)

        # timer for recording duration
        self._rec_timer = QtCore.QTimer(self); self._rec_timer.setSingleShot(True); self._rec_timer.timeout.connect(self.stop_recording)

    # ----------------------- UI helpers ---------------------------------------
    def _browse_out(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder", self.ed_out.text())
        if d:
            self.ed_out.setText(d)

    def _refresh_screens(self):
        self.cb_screens.clear()
        app = QtWidgets.QApplication.instance()
        for i, scr in enumerate(app.screens()):
            geo = scr.geometry()
            self.cb_screens.addItem(f"{i}: {geo.width()}x{geo.height()}", i)

    # -------------------- Devices / workers -----------------------------------
    def refresh_devices(self):
        print("[Main] Refreshing devices…")
        # Keep a persistent PySpin System; DO NOT ReleaseInstance here.
        devices: List[DeviceInfo] = []
        # PySpin first
        if HAS_PYSPIN:
            try:
                psp = PySpinCam.enumerate()
                for cam in psp:
                    devices.append(DeviceInfo("PySpin", cam.serial, f"PySpin [{cam.serial}]"))
            except Exception as e:
                print(f"[PySpin] enumerate error: {e}")
        # OpenCV device indices (0..3)
        if HAS_CV2:
            try:
                ocv = OpenCVCam.enumerate(max_indices=4)
                for oc in ocv:
                    devices.append(DeviceInfo("OpenCV", str(oc.index), f"OpenCV idx {oc.index}"))
            except Exception as e:
                print(f"[OpenCV] enumerate error: {e}")
        self.devices = devices

        # populate dropdowns
        def fill(cb: QtWidgets.QComboBox):
            cb.clear()
            for d in self.devices:
                cb.addItem(d.display, (d.backend, d.ident))
            if cb.count() == 0:
                cb.addItem("(none)", ("", ""))
        fill(self.cb_cam0)
        fill(self.cb_cam1)
        print(f"[Main] Found {len(self.devices)} devices.")

    def open_cam(self, cam_id: int):
        cb = self.cb_cam0 if cam_id == 0 else self.cb_cam1
        data = cb.currentData()
        if not data:
            QtWidgets.QMessageBox.warning(self, "No device", "Please select a camera device first.")
            return
        backend, ident = data
        device = DeviceInfo(backend, ident, f"{backend} {ident}")
        # Create or restart worker
        w = self.workers[cam_id]
        if w and w.isRunning():
            w.running = False
            w.wait(1000)
        w = CameraWorker(cam_id, device, self)
        w.newFrame.connect(self._on_frame)
        w.fpsUpdate.connect(self._on_fps)
        self.workers[cam_id] = w
        w.start()
        print(f"[Main] Camera {cam_id} worker started: {backend} {ident}")

    def toggle_preview(self, cam_id: int):
        # Only show labels when toggled on
        self.preview_on[cam_id] = not self.preview_on[cam_id]
        if cam_id == 0:
            self.lbl0.setVisible(self.preview_on[0])
        else:
            self.lbl1.setVisible(self.preview_on[1])
        print(f"[Main] Preview {cam_id} → {'ON' if self.preview_on[cam_id] else 'OFF'}")

    @QtCore.pyqtSlot(int, object)
    def _on_frame(self, cam_id: int, frame):
        # Paint preview only if toggled on
        if not self.preview_on[cam_id]:
            return
        # Convert BGR -> RGB and QImage
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        bytes_per_line = 3 * w
        # Using bytes() avoids memoryview TypeError on some PyQt builds
        qimg = QtGui.QImage(bytes(img.data), w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pm = QtGui.QPixmap.fromImage(qimg).scaled(400, 300, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        if cam_id == 0:
            self.lbl0.setPixmap(pm)
        else:
            self.lbl1.setPixmap(pm)

    @QtCore.pyqtSlot(int, float)
    def _on_fps(self, cam_id: int, fps: float):
        if cam_id == 0:
            self.lbl_fps0.setText(f"FPS0: {fps:.1f}")
        else:
            self.lbl_fps1.setText(f"FPS1: {fps:.1f}")

    # ------------------------ Recording ---------------------------------------
    def record_both(self):
        outdir = pathlib.Path(self.ed_out.text()).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        # Start writers on running workers
        ts = time.strftime("%Y%m%d_%H%M%S")
        started_any = False
        for cam_id in (0,1):
            w = self.workers[cam_id]
            if w and w.isRunning():
                path = outdir / f"cam{cam_id}_{ts}.avi"
                ok = w.start_recording(str(path), fps_hint=180.0)
                if ok:
                    started_any = True
        if started_any:
            self._rec_timer.start(int(self.ed_dur.value()*1000))
            print("[Main] Recording started.")
        else:
            QtWidgets.QMessageBox.warning(self, "No active cameras", "Start/open cameras first, then record.")

    def stop_recording(self):
        for cam_id in (0,1):
            w = self.workers[cam_id]
            if w and w.isRunning():
                w.stop_recording()
        print("[Main] Recording stopped (timer).")

    # ------------------------ Stimulus ----------------------------------------
    def show_stimulus(self):
        idx = self.cb_screens.currentData()
        fullscreen = self.chk_full.isChecked()
        app = QtWidgets.QApplication.instance()
        screens = app.screens()
        screen = screens[idx] if 0 <= idx < len(screens) else app.primaryScreen()
        dlg = StimulusWindow(screen, fullscreen, self)
        dlg.show()

    # ------------------------ Cleanup -----------------------------------------
    def closeEvent(self, e: QtGui.QCloseEvent):
        for w in self.workers:
            if w and w.isRunning():
                w.running = False
                w.wait(1000)
        super().closeEvent(e)

# --------------------------------- Entry --------------------------------------
def main():
    banner()
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())
