#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlyAPI.py (v1.40.0)
- Camera device refresh button
- Dropdown selectors per camera
- Toggleable, pop-out previews (OFF by default to save memory)
- High-FPS capture threads (only run when preview ON or during recording)
- Recording writes full source resolution
- Looming stimulus: black dot on white background, user sets total time (seconds)
- Dropdown to choose which monitor shows the stimulus + optional fullscreen
This file is self-contained and does not depend on previous broken builds.
"""

import os, sys, time, threading, queue, math, pathlib
from dataclasses import dataclass, field

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

APP_NAME = "FlyPy"
APP_VERSION = "1.40.0"

# ------------------------ Camera helpers ------------------------

def list_opencv_cameras(max_probe: int = 10):
    """Probe available camera indices conservatively. Returns list of (index, name)."""
    devices = []
    for idx in range(max_probe):
        cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
        ok = cap.isOpened()
        if not ok:
            cap.release()
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # try DSHOW too
            ok = cap.isOpened()
        if ok:
            # try to get a friendly name if possible
            name = f"OpenCV #{idx}"
            # resolution preview
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if w and h:
                name += f" ({w}x{h})"
            devices.append((idx, name))
            cap.release()
        else:
            # don't print warnings; just silently skip
            pass
    return devices

class CaptureThread(QtCore.QObject):
    """Pull frames from a cv2.VideoCapture in a worker thread with high FPS."""
    frame = QtCore.pyqtSignal(np.ndarray, float)  # (frame, timestamp)
    stopped = QtCore.pyqtSignal()

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self._stop = threading.Event()
        self._thread = None
        self.cap = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._thread = None
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.stopped.emit()

    def _open(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return None
        # Try to maximize FPS: many industrial cameras ignore these hints, but harmless
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 120)
        return cap

    def _run(self):
        self.cap = self._open()
        if self.cap is None:
            self.stopped.emit()
            return
        t_last = time.perf_counter()
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            t_now = time.perf_counter()
            if not ok:
                # brief backoff to avoid pegging CPU on disconnect
                time.sleep(0.005)
                continue
            self.frame.emit(frame, t_now)
            t_last = t_now
        # cleanup occurs in stop()

# ------------------------ Recorder ------------------------

class Recorder(QtCore.QObject):
    """Record frames from multiple CaptureThread sources at native resolution."""
    finished = QtCore.pyqtSignal(str)  # output dir

    def __init__(self, out_dir: str, parent=None):
        super().__init__(parent)
        self.out_dir = out_dir
        self.writers = {}   # cam_key -> (VideoWriter, is_color)
        self.running = False
        self._lock = threading.Lock()

    def _open_writer(self, cam_key: str, frame: np.ndarray, fps: float):
        os.makedirs(self.out_dir, exist_ok=True)
        h, w = frame.shape[:2]
        is_color = (frame.ndim == 3 and frame.shape[2] == 3)
        # MJPG works broadly and avoids channel mismatch warnings
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        pth = os.path.join(self.out_dir, f"{cam_key}.avi")
        writer = cv2.VideoWriter(pth, fourcc, max(fps, 1.0), (w, h), isColor=is_color)
        return writer, is_color

    def handle_frame(self, cam_key: str, frame: np.ndarray, fps_hint: float):
        with self._lock:
            if cam_key not in self.writers:
                vw, is_color = self._open_writer(cam_key, frame, fps_hint or 120.0)
                self.writers[cam_key] = (vw, is_color)
            else:
                vw, is_color = self.writers[cam_key]
            # Ensure frame matches writer color expectation
            if is_color and (frame.ndim == 2 or frame.shape[2] == 1):
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if (not is_color) and frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vw.write(frame)

    def stop(self):
        with self._lock:
            for vw, _ in self.writers.values():
                try: vw.release()
                except Exception: pass
            self.writers.clear()
        self.finished.emit(self.out_dir)

# ------------------------ Stimulus (looming dot) ------------------------

class LoomingStimulus(QtWidgets.QDialog):
    """
    Black dot on white background, radius grows from 0 to fill window
    over total_time_s seconds. Shows on a chosen monitor, optional fullscreen.
    """
    def __init__(self, screen: QtGui.QScreen, total_time_s: float, fullscreen: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Looming Stimulus")
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.total_time_s = max(0.001, float(total_time_s))
        self.t0 = time.perf_counter()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(1000//120)  # 120 Hz repaint target
        # place on screen
        geo = screen.geometry()
        self.setGeometry(geo)
        if fullscreen:
            # Move to screen and toggle fullscreen
            self.move(geo.topLeft())
            self.showFullScreen()
        else:
            self.show()

    def paintEvent(self, ev):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
        # background white
        qp.fillRect(self.rect(), QtCore.Qt.white)
        # radius grows linearly with time
        elapsed = time.perf_counter() - self.t0
        frac = min(1.0, elapsed / self.total_time_s)
        # Radius that covers min dimension at completion
        r = 0.5 * min(self.width(), self.height()) * frac
        cx, cy = self.width()//2, self.height()//2
        qp.setPen(QtCore.Qt.NoPen)
        qp.setBrush(QtGui.QBrush(QtCore.Qt.black))
        qp.drawEllipse(QtCore.QPoint(cx, cy), int(r), int(r))

# ------------------------ GUI ------------------------

@dataclass
class CamState:
    index: int = -1
    preview_on: bool = False
    fps: float = 0.0
    last_t: float = 0.0
    thread: CaptureThread = None

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1100, 700)

        self.camA = CamState()
        self.camB = CamState()
        self.recorder: Recorder | None = None
        self.recording = False

        self._build_ui()
        self._refresh_cameras()
        self._refresh_screens()

        # timers
        self.fps_timer = QtCore.QTimer(self)
        self.fps_timer.timeout.connect(self._update_fps_labels)
        self.fps_timer.start(250)

    # --------- UI building ---------
    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Row: camera controls
        row = QtWidgets.QHBoxLayout()
        self.btn_refresh = QtWidgets.QPushButton("Refresh Cameras")
        self.btn_refresh.clicked.connect(self._refresh_cameras)
        row.addWidget(self.btn_refresh)

        self.cb_camA = QtWidgets.QComboBox(); self.cb_camB = QtWidgets.QComboBox()
        row.addWidget(QtWidgets.QLabel("Cam A:")); row.addWidget(self.cb_camA)
        row.addWidget(QtWidgets.QLabel("Cam B:")); row.addWidget(self.cb_camB)

        self.chk_prevA = QtWidgets.QCheckBox("Preview A"); self.chk_prevB = QtWidgets.QCheckBox("Preview B")
        self.chk_prevA.stateChanged.connect(lambda _: self._toggle_preview('A'))
        self.chk_prevB.stateChanged.connect(lambda _: self._toggle_preview('B'))
        row.addWidget(self.chk_prevA); row.addWidget(self.chk_prevB)

        self.lbl_fpsA = QtWidgets.QLabel("FPS A: –"); self.lbl_fpsB = QtWidgets.QLabel("FPS B: –")
        row.addWidget(self.lbl_fpsA); row.addWidget(self.lbl_fpsB)
        row.addStretch(1)
        layout.addLayout(row)

        # Row: record controls
        recrow = QtWidgets.QHBoxLayout()
        self.le_outdir = QtWidgets.QLineEdit(str(pathlib.Path.cwd() / "FlyPy_Output"))
        recrow.addWidget(QtWidgets.QLabel("Output dir:")); recrow.addWidget(self.le_outdir, 1)
        self.btn_record = QtWidgets.QPushButton("Start Recording")
        self.btn_record.setCheckable(True)
        self.btn_record.toggled.connect(self._toggle_recording)
        recrow.addWidget(self.btn_record)
        layout.addLayout(recrow)

        # Collapsible previews (hidden to save memory until requested)
        self.grp_prevA = QtWidgets.QGroupBox("Camera A preview"); self.grp_prevA.setCheckable(True); self.grp_prevA.setChecked(False)
        self.grp_prevB = QtWidgets.QGroupBox("Camera B preview"); self.grp_prevB.setCheckable(True); self.grp_prevB.setChecked(False)
        self.grp_prevA.toggled.connect(lambda on: self._toggle_preview_group('A', on))
        self.grp_prevB.toggled.connect(lambda on: self._toggle_preview_group('B', on))

        glA = QtWidgets.QVBoxLayout(self.grp_prevA); glB = QtWidgets.QVBoxLayout(self.grp_prevB)
        self.lbl_prevA = QtWidgets.QLabel("(preview off)"); self.lbl_prevA.setMinimumHeight(220); self.lbl_prevA.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_prevB = QtWidgets.QLabel("(preview off)"); self.lbl_prevB.setMinimumHeight(220); self.lbl_prevB.setAlignment(QtCore.Qt.AlignCenter)
        glA.addWidget(self.lbl_prevA); glB.addWidget(self.lbl_prevB)

        layout.addWidget(self.grp_prevA)
        layout.addWidget(self.grp_prevB)

        # Stimulus controls
        stim = QtWidgets.QGroupBox("Looming stimulus (black dot on white)")
        sl = QtWidgets.QGridLayout(stim)
        self.sb_dot_time = QtWidgets.QDoubleSpinBox(); self.sb_dot_time.setRange(0.01, 120.0); self.sb_dot_time.setDecimals(3); self.sb_dot_time.setValue(1.0)
        self.cb_screens = QtWidgets.QComboBox()
        self.chk_fullscreen = QtWidgets.QCheckBox("Fullscreen")
        self.chk_fullscreen.setChecked(True)
        self.btn_stim = QtWidgets.QPushButton("Start stimulus")
        self.btn_stim.clicked.connect(self._start_stimulus)
        sl.addWidget(QtWidgets.QLabel("Dot total time (s):"), 0, 0); sl.addWidget(self.sb_dot_time, 0, 1)
        sl.addWidget(QtWidgets.QLabel("Output screen:"), 1, 0); sl.addWidget(self.cb_screens, 1, 1)
        sl.addWidget(self.chk_fullscreen, 2, 0)
        sl.addWidget(self.btn_stim, 2, 1)
        layout.addWidget(stim)

        layout.addStretch(1)

        # Status bar
        self.status = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.status)
        self._set_status("Ready. Previews are OFF by default.")

    def _set_status(self, msg: str):
        self.status.showMessage(msg)

    # --------- Camera handling ---------
    def _refresh_cameras(self):
        devs = list_opencv_cameras(max_probe=10)
        self.cb_camA.blockSignals(True); self.cb_camB.blockSignals(True)
        self.cb_camA.clear(); self.cb_camB.clear()
        for idx, name in devs:
            self.cb_camA.addItem(name, idx)
            self.cb_camB.addItem(name, idx)
        self.cb_camA.blockSignals(False); self.cb_camB.blockSignals(False)
        if devs:
            self.cb_camA.setCurrentIndex(0)
            if len(devs) > 1:
                self.cb_camB.setCurrentIndex(1)
        self._set_status(f"Detected cameras: {', '.join([n for _, n in devs]) or 'none'}")

    def _toggle_preview_group(self, which: str, on: bool):
        if which == 'A':
            self.chk_prevA.setChecked(on)
            if not on: self.lbl_prevA.setPixmap(QtGui.QPixmap())
        else:
            self.chk_prevB.setChecked(on)
            if not on: self.lbl_prevB.setPixmap(QtGui.QPixmap())

    def _toggle_preview(self, which: str):
        state = self.camA if which == 'A' else self.camB
        cb = self.cb_camA if which == 'A' else self.cb_camB
        group = self.grp_prevA if which == 'A' else self.grp_prevB
        label = self.lbl_prevA if which == 'A' else self.lbl_prevB

        want = (which == 'A' and self.chk_prevA.isChecked()) or (which == 'B' and self.chk_prevB.isChecked())
        # make sure group reflects the toggle
        group.blockSignals(True); group.setChecked(want); group.blockSignals(False)

        if want:
            # start thread
            index = cb.currentData()
            if index is None:
                self._set_status(f"No camera selected for {which}")
                return
            state.index = int(index)
            if state.thread:
                state.thread.stop()
            state.thread = CaptureThread(state.index)
            state.thread.frame.connect(lambda frame, t, w=which: self._on_frame(w, frame, t))
            state.thread.start()
            label.setText("Preview starting…")
        else:
            # stop thread
            if state.thread:
                state.thread.stop()
                state.thread = None
            label.setText("(preview off)")

    @QtCore.pyqtSlot(np.ndarray, float)
    def _on_frame(self, which: str, frame: np.ndarray, t: float):
        # FPS calculation
        state = self.camA if which == 'A' else self.camB
        if state.last_t:
            dt = t - state.last_t
            if dt > 0:
                # exponential moving average
                state.fps = 0.9*state.fps + 0.1*(1.0/dt) if state.fps else (1.0/dt)
        state.last_t = t

        # Only draw if preview box is visible (saves CPU)
        label = self.lbl_prevA if which == 'A' else self.lbl_prevB
        if (which == 'A' and self.grp_prevA.isChecked()) or (which == 'B' and self.grp_prevB.isChecked()):
            img = frame
            if img.ndim == 2:  # gray -> color for display
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            qimg = QtGui.QImage(img_rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg).scaled(label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            label.setPixmap(pix)

        # If recording, hand the frame to recorder
        if self.recording and self.recorder:
            key = f"cam_{which}"
            self.recorder.handle_frame(key, frame, state.fps or 120.0)

    def _update_fps_labels(self):
        a = f"{self.camA.fps:0.1f}" if self.camA.fps else "–"
        b = f"{self.camB.fps:0.1f}" if self.camB.fps else "–"
        self.lbl_fpsA.setText(f"FPS A: {a}")
        self.lbl_fpsB.setText(f"FPS B: {b}")

    def _toggle_recording(self, on: bool):
        if on:
            outdir = self.le_outdir.text().strip()
            if not outdir:
                QtWidgets.QMessageBox.warning(self, APP_NAME, "Please choose an output directory.")
                self.btn_record.setChecked(False); return
            self.recorder = Recorder(outdir)
            self.recording = True
            self.btn_record.setText("Stop Recording")
            self._set_status("Recording…")
        else:
            if self.recorder:
                self.recorder.stop()
                self.recorder = None
            self.recording = False
            self.btn_record.setText("Start Recording")
            self._set_status("Recording stopped.")

    # --------- Stimulus handling ---------
    def _refresh_screens(self):
        self.cb_screens.clear()
        for i, s in enumerate(QtWidgets.QApplication.screens()):
            g = s.geometry()
            self.cb_screens.addItem(f"Screen {i+1}: {g.width()}×{g.height()}", i)

    def _start_stimulus(self):
        idx = self.cb_screens.currentData()
        if idx is None:
            QtWidgets.QMessageBox.warning(self, APP_NAME, "No output screen available.")
            return
        scr = QtWidgets.QApplication.screens()[int(idx)]
        t_total = float(self.sb_dot_time.value())
        fs = self.chk_fullscreen.isChecked()
        dlg = LoomingStimulus(scr, t_total, fs, self)
        dlg.show()

# ------------------------ main ------------------------

def main():
    os.environ.setdefault('QT_ENABLE_HIGHDPI_SCALING','1')
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())
