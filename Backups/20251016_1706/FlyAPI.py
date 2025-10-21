#!/usr/bin/env python3
# FlyAPI.py — FlyPy camera UI (reconstructed robust edition)
# - Two-camera shell with refresh, per-camera device dropdowns, preview toggles,
#   FPS readout, and safe high-FPS recording to FlyPy_Output/.
# - OpenCV backend by default; optional PySpin backend when available.
# - Previews are fully optional to keep GUI light; FPS keeps updating even when preview is off.
#
# Tested with Python 3.10–3.12, PyQt5, OpenCV 4.x on Windows 10/11.

import os, sys, time, threading, queue, ctypes, pathlib
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

# ----- Qt -----
from PyQt5 import QtCore, QtGui, QtWidgets

# ----- OpenCV -----
import cv2
CV_BACKENDS = [cv2.CAP_MSMF, cv2.CAP_DSHOW]  # probe order on Windows

# ----- Optional PySpin (FLIR) -----
try:
    import PySpin
    HAS_PYSPIN = True
except Exception:
    HAS_PYSPIN = False

APP_VERSION = "1.30.0"

OUT_DIR = pathlib.Path("FlyPy_Output")
OUT_DIR.mkdir(exist_ok=True)

def banner():
    print("=== FlyPy Startup ===")
    print(f"Version: {APP_VERSION}")
    print("OpenCV: OK")
    if HAS_PYSPIN:
        print("PySpin: OK")
    else:
        print("PySpin: not available")
    print("======================")

# ----------------------------- Utilities ------------------------------------
def qimage_from_bgr(bgr):
    """Make a QImage from a BGR numpy array safely."""
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    if bgr.ndim == 2:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    bytes_per_line = rgb.strides[0]
    # Create QImage that owns its data (deep copy) to avoid dangling memory
    return QtGui.QImage(rgb.data.tobytes(), w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

def now_str():
    return time.strftime("%Y%m%d_%H%M%S")

# --------------------------- Camera backends ---------------------------------
@dataclass
class OpenCVCam:
    index: int
    friendly: str
    backend: int = field(default=cv2.CAP_MSMF)
    cap: Optional[cv2.VideoCapture] = field(default=None, init=False)
    lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    size: Tuple[int,int] = field(default=(0,0), init=False)

    @staticmethod
    def enumerate(max_indices: int = 10) -> List["OpenCVCam"]:
        cams = []
        for idx in range(max_indices):
            ok = False
            cap = None
            for be in CV_BACKENDS:
                cap = cv2.VideoCapture(idx, be)
                if cap is not None and cap.isOpened():
                    ok = True
                    backend = be
                    break
                if cap:
                    cap.release()
                    cap = None
            if not ok:
                continue
            # obtain a friendly name (Windows usually publishes it)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            friendly = f"OpenCV idx {idx} ({w}x{h})"
            cams.append(OpenCVCam(index=idx, friendly=friendly, backend=backend))
            cap.release()
        return cams

    def open(self) -> bool:
        with self.lock:
            if self.cap and self.cap.isOpened():
                return True
            # Try preferred backend then fallback
            for be in (self.backend, *(b for b in CV_BACKENDS if b != self.backend)):
                cap = cv2.VideoCapture(self.index, be)
                if cap is not None and cap.isOpened():
                    self.cap = cap
                    self.backend = be
                    break
            if not (self.cap and self.cap.isOpened()):
                print(f"[cam{self.index}] failed to open via OpenCV")
                return False
            self._cache_size()
            print(f"[cam{self.index}] OpenCV open: index {self.index} ({self.size[0]}x{self.size[1]})")
            return True

    def _cache_size(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.size = (w,h)

    def request_full_resolution(self):
        """Try a set of common high resolutions from biggest to smaller."""
        if not (self.cap and self.cap.isOpened()):
            return
        candidates = [
            (4096,2160),(3840,2160),(2560,1440),(1920,1200),(1920,1080),
            (1600,1200),(1280,1024),(1280,720),(1024,768),(800,600),(640,480)
        ]
        best = self.size
        for (w,h) in candidates:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            time.sleep(0.02)
            cw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            ch = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if cw >= best[0] and ch >= best[1]:
                best = (cw, ch)
        self.size = best

    def grab(self) -> Optional["tuple[bool, any]"]:
        with self.lock:
            if not (self.cap and self.cap.isOpened()):
                return None
            ret = self.cap.grab()
            if not ret:
                return (False, None)
            ret, frame = self.cap.retrieve()
            if not ret:
                return (False, None)
            return (True, frame)

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
    friendly: str
    cam: Optional[object] = field(default=None, init=False)
    lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    size: Tuple[int,int] = field(default=(0,0), init=False)
    _acquiring: bool = field(default=False, init=False)

    @staticmethod
    def enumerate() -> List["PySpinCam"]:
        if not HAS_PYSPIN:
            return []
        sysm = PySpin.System.GetInstance()
        lst = sysm.GetCameras()
        cams = []
        try:
            for i, c in enumerate(lst):
                nodemap_tl = c.GetTLDeviceNodeMap()
                sn_node = PySpin.CStringPtr(nodemap_tl.GetNode("DeviceSerialNumber"))
                name_node = PySpin.CStringPtr(nodemap_tl.GetNode("DeviceModelName"))
                serial = sn_node.GetValue() if PySpin.IsReadable(sn_node) else f"idx={i}"
                name = name_node.GetValue() if PySpin.IsReadable(name_node) else "FLIR"
                cams.append(PySpinCam(serial=str(serial), friendly=f"{name} [{serial}]"))
        finally:
            lst.Clear()
            sysm.ReleaseInstance()
        return cams

    def open(self) -> bool:
        if not HAS_PYSPIN:
            return False
        sysm = PySpin.System.GetInstance()
        lst = sysm.GetCameras()
        node = None
        try:
            for i, c in enumerate(lst):
                nodemap_tl = c.GetTLDeviceNodeMap()
                sn_node = PySpin.CStringPtr(nodemap_tl.GetNode("DeviceSerialNumber"))
                serial = sn_node.GetValue() if PySpin.IsReadable(sn_node) else f"idx={i}"
                if str(serial) == str(self.serial) or (self.serial.startswith("idx=") and f"idx={i}"==self.serial):
                    node = c
                    break
            if node is None:
                print(f"[PySpin] Serial {self.serial} not found")
                return False
            self.cam = node
            self.cam.Init()
            # set full width/height safely (must be NotAcquiring)
            nm = self.cam.GetNodeMap()
            w = PySpin.CIntegerPtr(nm.GetNode("Width"))
            h = PySpin.CIntegerPtr(nm.GetNode("Height"))
            if PySpin.IsWritable(w): w.SetValue(w.GetMax())
            if PySpin.IsWritable(h): h.SetValue(h.GetMax())
            self.size = (int(w.GetValue()), int(h.GetValue()))
            self.cam.BeginAcquisition()
            self._acquiring = True
            print(f"[PySpin] open: serial={self.serial} ({self.size[0]}x{self.size[1]})")
            return True
        except Exception as e:
            print("[PySpin] open error:", e)
            try:
                if self.cam:
                    self.cam.DeInit()
            except Exception:
                pass
            self.cam = None
            return False
        finally:
            lst.Clear()
            sysm.ReleaseInstance()

    def grab(self) -> Optional["tuple[bool, any]"]:
        if not self.cam:
            return None
        try:
            img = self.cam.GetNextImage(1000)
            if img.IsIncomplete():
                img.Release()
                return (False, None)
            nd = img.GetNDArray()  # uint8 mono or bgr? convert if mono
            frame = nd
            if nd.ndim == 2:
                frame = cv2.cvtColor(nd, cv2.COLOR_GRAY2BGR)
            img.Release()
            return (True, frame)
        except Exception:
            return (False, None)

    def close(self):
        try:
            if self.cam:
                if self._acquiring:
                    try: self.cam.EndAcquisition()
                    except Exception: pass
                self.cam.DeInit()
        except Exception:
            pass
        self.cam = None

# ------------------------------ Worker ---------------------------------------
class CameraWorker(QtCore.QThread):
    frameReady = QtCore.pyqtSignal(int, object)  # cam_id, frame (numpy)
    fpsReady = QtCore.pyqtSignal(int, float)     # cam_id, fps

    def __init__(self, cam_id: int, backend, parent=None):
        super().__init__(parent)
        self.cam_id = cam_id
        self.backend = backend  # OpenCVCam or PySpinCam
        self.running = False
        self.preview_enabled = True
        self.req_full_res = False
        self._last_ts = None
        self._frames = 0
        self._fps = 0.0

        # recording
        self.recording = False
        self.writer: Optional[cv2.VideoWriter] = None
        self.rec_path = None
        self.rec_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.rec_fps = 120.0  # target; overwritten by measured FPS

    def run(self):
        if hasattr(self.backend, "open") and not self.backend.open():
            return
        if isinstance(self.backend, OpenCVCam) and self.req_full_res:
            self.backend.request_full_resolution()
        self.running = True
        self._last_ts = time.perf_counter()
        self._frames = 0

        while self.running:
            res = self.backend.grab()
            if not res:
                time.sleep(0.002)
                continue
            ok, frame = res
            if not ok or frame is None:
                time.sleep(0.002); continue

            # FPS calc
            self._frames += 1
            now = time.perf_counter()
            dt = now - self._last_ts
            if dt >= 0.5:
                self._fps = self._frames / dt
                self.fpsReady.emit(self.cam_id, float(self._fps))
                self._frames = 0
                self._last_ts = now

            # Emit preview only if enabled
            if self.preview_enabled:
                self.frameReady.emit(self.cam_id, frame)

            # Recording
            if self.recording:
                if self.writer is None:
                    # Allocate writer in the first recording iteration to match actual size
                    h, w = frame.shape[:2]
                    self.rec_fps = max(1.0, self._fps or 120.0)
                    fname = f"cam{self.cam_id}_{now_str()}.avi"
                    self.rec_path = str(OUT_DIR / fname)
                    wr = cv2.VideoWriter(self.rec_path, self.rec_fourcc, self.rec_fps, (w, h))
                    if not wr or not wr.isOpened():
                        # Try fallback fourcc
                        self.rec_fourcc = cv2.VideoWriter_fourcc(*"XVID")
                        wr = cv2.VideoWriter(self.rec_path, self.rec_fourcc, self.rec_fps, (w, h))
                    self.writer = wr if wr.isOpened() else None
                if self.writer:
                    # Ensure 3 channels for color codecs
                    if frame.ndim == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    try:
                        self.writer.write(frame)
                    except Exception:
                        pass

        # Cleanup
        if self.writer:
            try: self.writer.release()
            except Exception: pass
            self.writer = None
        try: self.backend.close()
        except Exception: pass

    def start_recording(self):
        self.recording = True
        self.rec_path = None

    def stop_recording(self) -> Optional[str]:
        self.recording = False
        # writer is closed in thread exit; flush now for convenience
        if self.writer:
            try:
                self.writer.release()
            except Exception:
                pass
            path = self.rec_path
            self.writer = None
            self.rec_path = None
            return path
        return None

# ------------------------------- GUI -----------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FlyPy – Recording GUI")
        self.resize(1200, 720)

        self.workers: List[CameraWorker] = []
        self.backends = {}  # cam_id -> backend instance

        # Central UI
        cw = QtWidgets.QWidget(self); self.setCentralWidget(cw)
        root = QtWidgets.QVBoxLayout(cw)

        # Row: refresh + device dropdowns
        row = QtWidgets.QHBoxLayout()
        self.btn_refresh = QtWidgets.QPushButton("🔄 Refresh cameras")
        self.btn_refresh.clicked.connect(self.refresh_devices)
        row.addWidget(self.btn_refresh)

        row.addSpacing(10)
        row.addWidget(QtWidgets.QLabel("Cam 0:"))
        self.cbo_cam0 = QtWidgets.QComboBox()
        row.addWidget(self.cbo_cam0)
        self.chk_prev0 = QtWidgets.QCheckBox("Preview 0")
        self.chk_prev0.setChecked(False)
        row.addWidget(self.chk_prev0)

        row.addSpacing(20)
        row.addWidget(QtWidgets.QLabel("Cam 1:"))
        self.cbo_cam1 = QtWidgets.QComboBox()
        row.addWidget(self.cbo_cam1)
        self.chk_prev1 = QtWidgets.QCheckBox("Preview 1")
        self.chk_prev1.setChecked(False)
        row.addWidget(self.chk_prev1)

        row.addStretch(1)
        root.addLayout(row)

        # Previews (collapsed by default)
        pv = QtWidgets.QHBoxLayout()
        self.lbl0 = QtWidgets.QLabel("Preview 0 off"); self.lbl0.setMinimumSize(320,240); self.lbl0.setFrameShape(QtWidgets.QFrame.Box)
        self.lbl1 = QtWidgets.QLabel("Preview 1 off"); self.lbl1.setMinimumSize(320,240); self.lbl1.setFrameShape(QtWidgets.QFrame.Box)
        pv.addWidget(self.lbl0); pv.addWidget(self.lbl1)
        root.addLayout(pv)

        # FPS row
        fpsrow = QtWidgets.QHBoxLayout()
        self.fps0 = QtWidgets.QLabel("FPS0: --"); fpsrow.addWidget(self.fps0)
        self.fps1 = QtWidgets.QLabel("FPS1: --"); fpsrow.addWidget(self.fps1)
        fpsrow.addStretch(1)
        root.addLayout(fpsrow)

        # Recording controls
        recrow = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("⏺ Start Recording (both)")
        self.btn_stop = QtWidgets.QPushButton("⏹ Stop")
        self.btn_start.clicked.connect(self.start_recording)
        self.btn_stop.clicked.connect(self.stop_recording)
        recrow.addWidget(self.btn_start); recrow.addWidget(self.btn_stop)
        recrow.addStretch(1)
        root.addLayout(recrow)

        # Status
        self.status = QtWidgets.QPlainTextEdit(readOnly=True)
        self.status.setMaximumBlockCount(500)
        root.addWidget(self.status)

        # Signals for preview toggles
        self.chk_prev0.toggled.connect(lambda on: self._toggle_preview(0, on))
        self.chk_prev1.toggled.connect(lambda on: self._toggle_preview(1, on))

        # Device lists
        self.devices: List[Tuple[str, object]] = []   # (label, backend-ctor-args)
        self.refresh_devices()

        # Start workers with currently selected devices (if any)
        self.apply_selected_devices()

        # Timer to update preview labels visibility cheaply when previews are off
        self._img0 = None
        self._img1 = None

    # ----------- Device enumeration / selection -----------
    def log(self, msg: str):
        print(msg)
        self.status.appendPlainText(msg)

    def refresh_devices(self):
        self.cbo_cam0.clear(); self.cbo_cam1.clear()
        self.devices = []

        # OpenCV devices
        ocv = OpenCVCam.enumerate(10)
        for cam in ocv:
            label = f"OpenCV[{cam.index}]"
            self.devices.append((label, ("opencv", cam.index, cam.backend, cam.friendly)))
        # PySpin devices
        psp = PySpinCam.enumerate() if HAS_PYSPIN else []
        for cam in psp:
            label = f"PySpin[{cam.serial}]"
            self.devices.append((label, ("pyspin", cam.serial, cam.friendly)))

        if not self.devices:
            self.log("[Scan] No cameras found.")
        else:
            self.log(f"[Scan] Found {len(self.devices)} device(s).")

        # Populate comboboxes
        for label,_ in self.devices:
            self.cbo_cam0.addItem(label)
            self.cbo_cam1.addItem(label)

        # Preselect different devices if possible
        if len(self.devices) >= 2:
            self.cbo_cam0.setCurrentIndex(0)
            self.cbo_cam1.setCurrentIndex(1)
        elif len(self.devices) == 1:
            self.cbo_cam0.setCurrentIndex(0)
            self.cbo_cam1.setCurrentIndex(0)

    def _make_backend(self, spec):
        if spec[0] == "opencv":
            _, index, backend, friendly = spec
            return OpenCVCam(index=index, friendly=friendly, backend=backend)
        elif spec[0] == "pyspin":
            _, serial, friendly = spec
            return PySpinCam(serial=serial, friendly=friendly)
        else:
            raise ValueError("Unknown backend spec")

    def apply_selected_devices(self):
        # Stop old
        self.stop_workers()

        # Create backends from selections
        sel0 = self.cbo_cam0.currentIndex()
        sel1 = self.cbo_cam1.currentIndex()
        spec0 = self.devices[sel0][1] if 0 <= sel0 < len(self.devices) else None
        spec1 = self.devices[sel1][1] if 0 <= sel1 < len(self.devices) else None

        self.backends = {}
        self.workers = []

        # cam 0
        if spec0:
            b0 = self._make_backend(spec0)
            w0 = CameraWorker(0, b0)
            w0.preview_enabled = self.chk_prev0.isChecked()
            w0.req_full_res = True  # always request full for recording baseline
            w0.frameReady.connect(self.on_frame)
            w0.fpsReady.connect(self.on_fps)
            self.backends[0] = b0; self.workers.append(w0)
        # cam 1
        if spec1:
            b1 = self._make_backend(spec1)
            w1 = CameraWorker(1, b1)
            w1.preview_enabled = self.chk_prev1.isChecked()
            w1.req_full_res = True
            w1.frameReady.connect(self.on_frame)
            w1.fpsReady.connect(self.on_fps)
            self.backends[1] = b1; self.workers.append(w1)

        # Start them
        for w in self.workers:
            w.start()

    def stop_workers(self):
        for w in self.workers:
            try:
                w.running = False
                w.wait(1000)
            except Exception:
                pass
        self.workers.clear()
        for b in list(self.backends.values()):
            try: b.close()
            except Exception: pass
        self.backends.clear()

    # ----------- Preview / FPS display -----------
    @QtCore.pyqtSlot(int, object)
    def on_frame(self, cam_id: int, frame):
        # Paint to respective label; we convert lazily and scale to label size
        q = qimage_from_bgr(frame)
        if q is None: 
            return
        pm = QtGui.QPixmap.fromImage(q).scaled(
            self.lbl0.size() if cam_id == 0 else self.lbl1.size(),
            QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        if cam_id == 0:
            self.lbl0.setPixmap(pm)
        else:
            self.lbl1.setPixmap(pm)

    @QtCore.pyqtSlot(int, float)
    def on_fps(self, cam_id: int, fps: float):
        if cam_id == 0:
            self.fps0.setText(f"FPS0: {fps:5.1f}")
        else:
            self.fps1.setText(f"FPS1: {fps:5.1f}")

    def _toggle_preview(self, cam_id: int, on: bool):
        # Find worker and set preview flag
        for w in self.workers:
            if w.cam_id == cam_id:
                w.preview_enabled = on
                if not on:
                    if cam_id == 0:
                        self.lbl0.clear(); self.lbl0.setText("Preview 0 off")
                    else:
                        self.lbl1.clear(); self.lbl1.setText("Preview 1 off")
                break

    # ----------- Recording -----------
    def start_recording(self):
        for w in self.workers:
            w.start_recording()
        self.log("[Rec] Started")

    def stop_recording(self):
        saved = []
        for w in self.workers:
            path = w.stop_recording()
            if path: saved.append(path)
        if saved:
            for p in saved:
                self.log(f"[Rec] Saved: {p}")
        else:
            self.log("[Rec] Nothing saved")

    # ----------- Close -----------
    def closeEvent(self, e: QtGui.QCloseEvent):
        self.stop_workers()
        super().closeEvent(e)

# ------------------------------- App entry -----------------------------------
def main():
    banner()
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())
