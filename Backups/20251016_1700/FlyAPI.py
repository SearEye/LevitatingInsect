#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlyAPI.py — FlyPy Camera & Recording UI
Version: 1.27.0
- Adds per‑camera preview toggles (off by default) to reduce memory/GPU load.
- Refresh Cameras button + dropdowns populated on each refresh.
- Separate high‑FPS popout preview window.
- Recording writes full native resolution from each camera.
- Fixes QImage conversion crash (memoryview) by copying to bytes.
- Safe when only one camera is present; robust OpenCV back end with optional PySpin enumeration.

This is a self‑contained UI intended as a drop‑in replacement when the older
SettingsGUI is missing methods like set_preview_image(). It exposes a
compatible SettingsGUI object and set_preview_image(idx, img_bgr) method,
so external code that calls self.gui.set_preview_image(...) will not crash.

Tested against: Python 3.10, PyQt5 5.15, OpenCV 4.10+, Windows 10/11.
"""
import os, sys, time, threading, queue, datetime, math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any

os.environ.setdefault('QT_ENABLE_HIGHDPI_SCALING','1')
os.environ.setdefault('QT_SCALE_FACTOR','1')

try:
    from PyQt5 import QtWidgets, QtCore, QtGui
except Exception as e:
    print("[BOOT] PyQt5 import failed:", e)
    raise

import cv2
import numpy as np

APP_VER = "1.27.0"

# ---------- Helpers ----------

def banner():
    print("=== FlyPy Startup ===")
    print(f"Version: {APP_VER}")
    print(f"OpenCV: {'OK' if cv2.__version__ else 'not found'}")
    # PsychoPy optional in this UI
    print("PsychoPy: not available (OpenCV fallback)")
    # PySpin availability (for listing only)
    try:
        import PySpin  # noqa
        print("PySpin: OK")
    except Exception:
        print("PySpin: not available")
    print("======================")

def qimage_from_bgr(img_bgr: np.ndarray) -> QtGui.QImage:
    if img_bgr is None or img_bgr.size == 0:
        return QtGui.QImage()
    if len(img_bgr.shape) == 2:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    # convert to bytes to avoid memoryview TypeError on some PyQt builds
    buf = img_rgb.tobytes()
    qimg = QtGui.QImage(buf, w, h, 3*w, QtGui.QImage.Format_RGB888)
    return qimg

# ---------- Camera back end ----------

def list_opencv_devices(max_check=10) -> List[str]:
    names = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        ok = cap.isOpened()
        if not ok:
            # Try with DSHOW, but don't hold the device
            cap.release()
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            ok = cap.isOpened()
        if ok:
            names.append(f"OpenCV index {i}")
        cap.release()
    return names

def list_pyspin_devices() -> List[str]:
    try:
        import PySpin
        sysm = PySpin.System.GetInstance()
        lst = sysm.GetCameras()
        out = []
        for cam in lst:
            try:
                nodemap_tldevice = cam.GetTLDeviceNodeMap()
                serial_node = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
                model_node = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceModelName'))
                serial = serial_node.GetValue() if PySpin.IsReadable(serial_node) else "?"
                model  = model_node.GetValue()  if PySpin.IsReadable(model_node) else "FLIR"
                out.append(f"PySpin {model} S/N {serial}")
            except Exception:
                out.append("PySpin camera")
            finally:
                cam = None
        lst.Clear()
        sysm.ReleaseInstance()
        return out
    except Exception:
        return []

@dataclass
class OpenCVCam:
    ident: str  # e.g. "OpenCV index 0"
    index: int
    cap: Optional[cv2.VideoCapture] = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    latest: Optional[np.ndarray] = None
    running: bool = False
    fps: float = 0.0
    size: Tuple[int,int] = (0,0)

    def open(self) -> bool:
        with self.lock:
            if self.cap and self.cap.isOpened():
                return True
            # Prefer MSMF on modern Windows; if not, DSHOW
            self.cap = cv2.VideoCapture(self.index, cv2.CAP_MSMF)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                print(f"[cam{self.index}] failed to open")
                return False
            # Query native resolution
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            self.size = (w,h)
            print(f"[cam{self.index}] OpenCV open: index {self.index} ({w}x{h})")
            return True

    def close(self):
        with self.lock:
            if self.cap:
                self.cap.release()
            self.cap = None
            self.running = False

    def grab_loop(self, preview_enabled_fn):
        # Tight grab loop to keep FPS high whether or not preview is shown.
        self.running = True
        frame_count = 0
        t0 = time.time()
        while self.running:
            with self.lock:
                cap = self.cap
            if not cap or not cap.isOpened():
                time.sleep(0.01); continue
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.002); continue
            self.latest = frame
            frame_count += 1
            now = time.time()
            if now - t0 >= 0.5:
                self.fps = frame_count / (now - t0)
                frame_count = 0
                t0 = now
            # If preview disabled, sleep a touch to reduce GUI churn
            if not preview_enabled_fn():
                time.sleep(0.001)

    def start(self, preview_enabled_fn):
        if not self.open(): return
        if self.running: return
        th = threading.Thread(target=self.grab_loop, args=(preview_enabled_fn,), daemon=True)
        th.start()

    def read_latest(self) -> Tuple[Optional[np.ndarray], float]:
        # returns last frame and current fps
        return self.latest, self.fps

    def record(self, path: str, seconds: float, fourcc='XVID'):
        """Record at full native resolution."""
        if not self.open(): 
            raise RuntimeError("Camera not open")
        w,h = self.size
        # If size unknown, peek one frame
        if w<=0 or h<=0:
            frm, _ = self.read_latest()
            if frm is None:
                ok, frm = self.cap.read()
                if not ok: raise RuntimeError("No frame to size recording")
            h, w = frm.shape[:2]
            self.size=(w,h)
        code = cv2.VideoWriter_fourcc(*fourcc)
        # Ensure color flag matches frames
        is_color = True
        vw = cv2.VideoWriter(path, code, max(1.0, self.fps or 60.0), (w,h), isColor=is_color)
        if not vw.isOpened():
            raise RuntimeError("VideoWriter open failed")
        t_end = time.time() + max(0.1, seconds)
        while time.time() < t_end:
            frame, _ = self.read_latest()
            if frame is None:
                ok, frame = self.cap.read()
                if not ok: continue
            # Ensure 3 channels for writer
            if len(frame.shape)==2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2]==4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            vw.write(frame)
        vw.release()

# ---------- UI ----------

class PopoutPreview(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("High‑FPS Preview")
        self.lbl = QtWidgets.QLabel("No feed", self)
        self.lbl.setAlignment(QtCore.Qt.AlignCenter)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.lbl)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(15)  # ~66fps UI updates; grabbing stays in worker
        self.timer.timeout.connect(self._tick)
        self.cam: Optional[OpenCVCam] = None

    def show_for(self, cam: OpenCVCam):
        self.cam = cam
        self.timer.start()
        self.show()

    def _tick(self):
        if not self.cam: return
        frame, fps = self.cam.read_latest()
        if frame is None: return
        q = qimage_from_bgr(frame)
        if not q.isNull():
            self.lbl.setPixmap(QtGui.QPixmap.fromImage(q).scaled(self.lbl.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            self.setWindowTitle(f"High‑FPS Preview – {fps:.1f} fps")

class SettingsGUI(QtWidgets.QMainWindow):
    """Compatible replacement exposing set_preview_image(idx, img_bgr)."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FlyPy – Cameras")
        self.resize(1200, 700)
        self.preview_enabled = [False, False]
        self.cams: List[Optional[OpenCVCam]] = [None, None]

        # Widgets
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        grid = QtWidgets.QGridLayout(central)

        # Top bar: refresh + device dropdowns + preview toggles
        self.bt_refresh = QtWidgets.QPushButton("Refresh cameras")
        self.bt_refresh.clicked.connect(self.refresh_cameras)

        self.dd_cam0 = QtWidgets.QComboBox(); self.dd_cam1 = QtWidgets.QComboBox()
        self.dd_cam0.setMinimumWidth(280); self.dd_cam1.setMinimumWidth(280)

        self.chk_prev0 = QtWidgets.QCheckBox("Show Preview 0"); self.chk_prev1 = QtWidgets.QCheckBox("Show Preview 1")
        self.chk_prev0.setChecked(False); self.chk_prev1.setChecked(False)
        self.chk_prev0.toggled.connect(lambda v: self._toggle_preview(0, v))
        self.chk_prev1.toggled.connect(lambda v: self._toggle_preview(1, v))

        self.bt_pop0 = QtWidgets.QPushButton("Popout 0"); self.bt_pop1 = QtWidgets.QPushButton("Popout 1")
        self.bt_pop0.clicked.connect(lambda: self._open_popout(0))
        self.bt_pop1.clicked.connect(lambda: self._open_popout(1))
        self.pop = PopoutPreview(self)

        # Previews (hidden until toggled on)
        self.preview0 = QtWidgets.QLabel("Preview 0 OFF"); self.preview0.setObjectName("preview0")
        self.preview1 = QtWidgets.QLabel("Preview 1 OFF"); self.preview1.setObjectName("preview1")
        for lbl in (self.preview0, self.preview1):
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setMinimumHeight(280)
            lbl.setFrameShape(QtWidgets.QFrame.StyledPanel)

        # FPS labels
        self.lbl_fps0 = QtWidgets.QLabel("0.0 fps"); self.lbl_fps1 = QtWidgets.QLabel("0.0 fps")

        # Recording controls (full‑res)
        self.bt_rec0 = QtWidgets.QPushButton("Record 0 (5s)")
        self.bt_rec1 = QtWidgets.QPushButton("Record 1 (5s)")
        self.bt_rec0.clicked.connect(lambda: self._record(0))
        self.bt_rec1.clicked.connect(lambda: self._record(1))

        # Layout
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.bt_refresh)
        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("Cam0:")); top.addWidget(self.dd_cam0)
        top.addWidget(self.chk_prev0); top.addWidget(self.bt_pop0)
        top.addSpacing(24)
        top.addWidget(QtWidgets.QLabel("Cam1:")); top.addWidget(self.dd_cam1)
        top.addWidget(self.chk_prev1); top.addWidget(self.bt_pop1)
        top.addStretch(1)

        grid.addLayout(top, 0, 0, 1, 2)
        grid.addWidget(self.preview0, 1, 0)
        grid.addWidget(self.preview1, 1, 1)
        grid.addWidget(self.lbl_fps0, 2, 0, alignment=QtCore.Qt.AlignLeft)
        grid.addWidget(self.lbl_fps1, 2, 1, alignment=QtCore.Qt.AlignLeft)
        recbar = QtWidgets.QHBoxLayout()
        recbar.addWidget(self.bt_rec0); recbar.addWidget(self.bt_rec1); recbar.addStretch(1)
        grid.addLayout(recbar, 3, 0, 1, 2)

        # Timer to update FPS labels and keep UI responsive
        self.timer = QtCore.QTimer(self); self.timer.setInterval(200)
        self.timer.timeout.connect(self._refresh_fps)
        self.timer.start()

        self.refresh_cameras()

    # ---- External compatibility API ----
    def set_preview_image(self, idx: int, img_bgr: np.ndarray):
        """Safe to call regardless of preview toggle state."""
        if idx not in (0,1): return
        lbl = self.preview0 if idx==0 else self.preview1
        if not self.preview_enabled[idx]:
            # Do not render image when preview is OFF (saves CPU/GPU).
            return
        qimg = qimage_from_bgr(img_bgr)
        if qimg.isNull(): return
        lbl.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(lbl.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    # ---- Internal UI plumbing ----
    def refresh_cameras(self):
        opencv = list_opencv_devices(10)
        pyspin = list_pyspin_devices()
        # Remember selection
        sel0 = self.dd_cam0.currentText()
        sel1 = self.dd_cam1.currentText()
        self.dd_cam0.clear(); self.dd_cam1.clear()
        # Populate OpenCV first (directly usable)
        for name in opencv:
            self.dd_cam0.addItem(name); self.dd_cam1.addItem(name)
        # Then PySpin (listing only in this lightweight UI)
        for name in pyspin:
            self.dd_cam0.addItem(name); self.dd_cam1.addItem(name)
        # Restore selection when possible
        for dd, prev in ((self.dd_cam0, sel0),(self.dd_cam1, sel1)):
            ix = dd.findText(prev) if prev else -1
            if ix >= 0: dd.setCurrentIndex(ix)
        # (Re)open OpenCV devices
        self._ensure_cam(0); self._ensure_cam(1)

    def _ensure_cam(self, idx:int):
        dd = self.dd_cam0 if idx==0 else self.dd_cam1
        txt = dd.currentText()
        if txt.startswith("OpenCV index"):
            index = int(txt.split()[-1])
            cam = OpenCVCam(txt, index)
            cam.start(lambda i=idx: self.preview_enabled[i])
            self.cams[idx] = cam
        else:
            # For now, unsupported in this lightweight script; set None
            self.cams[idx] = None

    def _toggle_preview(self, idx:int, value:bool):
        self.preview_enabled[idx] = bool(value)
        lbl = self.preview0 if idx==0 else self.preview1
        if not value:
            lbl.setText(f"Preview {idx} OFF")
            lbl.setPixmap(QtGui.QPixmap())
        else:
            lbl.setText("")

    def _open_popout(self, idx:int):
        cam = self.cams[idx]
        if cam is None:
            QtWidgets.QMessageBox.warning(self, "No camera", f"Camera {idx} not available.")
            return
        self.pop.show_for(cam)

    def _refresh_fps(self):
        for i in (0,1):
            cam = self.cams[i]
            fps = cam.fps if cam else 0.0
            (self.lbl_fps0 if i==0 else self.lbl_fps1).setText(f"{fps:.1f} fps")
            # If preview is on, draw most recent frame
            if self.preview_enabled[i] and cam:
                frm, _ = cam.read_latest()
                if frm is not None:
                    self.set_preview_image(i, frm)

    def _record(self, idx:int):
        cam = self.cams[idx]
        if not cam:
            QtWidgets.QMessageBox.warning(self, "No camera", f"Camera {idx} not available.")
            return
        out_dir = os.path.join(os.getcwd(), "FlyPy_Output")
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"cam{idx}_{stamp}.avi")
        try:
            cam.record(path, seconds=5.0, fourcc='XVID')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Record failed", str(e))
            return
        QtWidgets.QMessageBox.information(self, "Saved", f"Camera {idx} clip saved:\n{path}")

# ---------- App ----------

def main():
    banner()
    app = QtWidgets.QApplication(sys.argv)
    ui = SettingsGUI()
    ui.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())
