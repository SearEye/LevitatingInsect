
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlyAPI.py — FlyPy full GUI (cameras + recording)
v1.30.0

- Restores full recording GUI while keeping the robust camera input code you have now.
- Adds "Refresh Cameras" button and dropdowns for both PySpin and OpenCV devices per camera.
- Live, high-FPS previews open in separate windows (non-blocking).
- Recording reworked to avoid FFmpeg "expected 1 channel but got 3" by always writing BGR.
- Concurrent recording from both cameras with status in the UI; per-camera quick record also available.
- Outputs saved under FlyPy_Output/<timestamp>/camX_<timestamp>.mp4 + JSON metadata.
"""

import os, sys, time, threading, datetime, traceback, json
os.environ.setdefault('QT_ENABLE_HIGHDPI_SCALING','1')
os.environ.setdefault('QT_SCALE_FACTOR','1')
os.environ.setdefault('QT_OPENGL','software')
os.environ.setdefault('OPENCV_VIDEOIO_DEBUG','0')
os.environ.setdefault('OPENCV_LOG_LEVEL','ERROR')

from typing import List, Dict, Optional, Tuple

APP_VERSION = "1.30.0"

def ts_now():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def safe_print(*a, **k):
    try:
        print(*a, **k, flush=True)
    except Exception:
        pass

# ---- OpenCV availability ----
try:
    import cv2
    OPENCV_OK = True
    # Silence OpenCV backend warnings if available
    try:
        from cv2 import utils as cv2utils
        try:
            cv2utils.logging.setLogLevel(cv2utils.logging.LOG_LEVEL_SILENT)
        except Exception:
            pass
    except Exception:
        pass
except Exception:
    cv2 = None
    OPENCV_OK = False

# ---- PySpin availability ----
PYSPIN_OK = False
try:
    import PySpin  # type: ignore
    PYSPIN_OK = True
except Exception:
    pass

from PyQt5 import QtCore, QtGui, QtWidgets

# ----------------- Global PySpin System (singleton) -----------------
class _SpinSystemHolder:
    sys = None
    @classmethod
    def get(cls):
        if not PYSPIN_OK:
            return None
        if cls.sys is None:
            cls.sys = PySpin.System.GetInstance()
        return cls.sys
    @classmethod
    def release(cls):
        if cls.sys is not None:
            try:
                cls.sys.ReleaseInstance()
            except Exception as e:
                safe_print("[PySpin] Release error:", e)
            cls.sys = None

# ----------------- Enumeration helpers -----------------
def enum_opencv() -> List[Dict]:
    out = []
    if not OPENCV_OK:
        return out
    # On Windows, try MSMF first then DSHOW; only accept truly opened devices
    caps = []
    try:
        caps.append(getattr(cv2, 'CAP_MSMF'))
    except Exception:
        pass
    try:
        caps.append(getattr(cv2, 'CAP_DSHOW'))
    except Exception:
        pass
    if not caps:
        caps = [getattr(cv2, 'CAP_ANY', 0)]
    seen = set()
    # Probe a small range to avoid noisy logs
    for i in range(0, 6):
        cap = None
        opened = False
        for backend in caps:
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap is not None and cap.isOpened():
                    opened = True
                    break
                else:
                    if cap is not None:
                        cap.release()
                        cap = None
            except Exception:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                cap = None
        if not opened or cap is None:
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        ident = f"index:{i}"
        if ident in seen:
            continue
        seen.add(ident)
        out.append({"index": i, "label": f"index:{i} ({w}x{h})", "ident": ident})
    return out

def _get_str(nodemap, name: str) -> str:
    try:
        node = PySpin.CStringPtr(nodemap.GetNode(name))
        if PySpin.IsAvailable(node) and PySpin.IsReadable(node):
            return node.GetValue()
    except Exception:
        pass
    return ""

POSSIBLE_INSTANCE_NAMES = [
    "DeviceInstanceID", "DeviceInstanceId", "DeviceLocation", "DeviceID",
    "DeviceGuid", "DeviceGUID", "DeviceAddress", "DeviceDisplayName"
]

def enum_pyspin() -> List[Dict]:
    res: List[Dict] = []
    if not PYSPIN_OK:
        return res
    sysm = _SpinSystemHolder.get()
    if sysm is None:
        return res
    cam_list = sysm.GetCameras()
    try:
        n = cam_list.GetSize()
        for i in range(n):
            cam = cam_list.GetByIndex(i)
            tl = cam.GetTLDeviceNodeMap()
            serial = _get_str(tl, "DeviceSerialNumber")
            model  = _get_str(tl, "DeviceModelName")
            vendor = _get_str(tl, "DeviceVendorName")
            iface  = _get_str(tl, "InterfaceDisplayName") or _get_str(tl, "DeviceVendorName")
            inst = ""
            for nm in POSSIBLE_INSTANCE_NAMES:
                inst = _get_str(tl, nm)
                if inst:
                    break
            short_inst = inst if not inst or len(inst) <= 54 else ("…" + inst[-54:])
            label = f"idx:{i} — {model or 'Unknown'} SN {serial or 'NA'} | {iface or 'Interface'} | {short_inst or 'no-InstanceID'}"
            ident = f"serial:{serial}" if serial else (f"inst:{inst}" if inst else f"idx:{i}")
            res.append({
                "index": i,
                "serial": serial,
                "model": model,
                "vendor": vendor,
                "iface": iface,
                "instance": inst,
                "label": label,
                "ident": ident,
            })
        res.sort(key=lambda d:(d.get("instance",""), d.get("serial",""), d.get("index",0)))
        return res
    finally:
        cam_list.Clear()
        del cam_list

# ----------------- Camera backends -----------------
class BaseCamera:
    def __init__(self, ident: str):
        self.ident = ident
        self.opened = False
        self._streaming = False
    def open(self) -> bool: raise NotImplementedError
    def close(self): raise NotImplementedError
    def start(self)->bool:
        self._streaming = True; return True
    def stop(self): self._streaming = False
    def read(self): raise NotImplementedError
    def record_clip(self, out_path: str, duration_s: float, fourcc: str = "mp4v", fps_hint: float = 120.0):
        """
        Simple inline recorder (used by Quick Record buttons). For full recording from both cameras,
        the Recorder class below is used.
        Always writes BGR frames and opens writer with isColor=True to avoid FFmpeg warnings.
        """
        import numpy as np
        ensure_dir(os.path.dirname(out_path))
        first = self.read()
        if first is None:
            raise RuntimeError("No frame available to start recording")
        _, fr0 = first
        h, w = fr0.shape[:2]
        # Always write color (BGR)
        if fr0.ndim==2:
            fr0 = cv2.cvtColor(fr0, cv2.COLOR_GRAY2BGR)
        # Choose extension
        ext = ".mp4" if fourcc.lower()!="mjpg" else ".avi"
        if not out_path.lower().endswith((".mp4",".avi")):
            out_path += ext
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc.upper()), fps_hint, (w,h), isColor=True)
        if not vw.isOpened(): raise RuntimeError(f"Failed to open writer for {out_path}")
        t0 = time.time(); frames=0
        # write first frame
        vw.write(fr0); frames+=1
        while time.time()-t0 < duration_s:
            f = self.read()
            if f is None:
                continue
            _, fr = f
            if fr.ndim==2:
                fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
            vw.write(fr); frames+=1
        vw.release()
        dt=max(1e-3,time.time()-t0)
        return {"path":out_path, "frames":frames, "fps":frames/dt}

class OpenCVCamera(BaseCamera):
    def __init__(self, ident:str):
        super().__init__(ident); self.cap=None
    def open(self)->bool:
        if not OPENCV_OK: return False
        try:
            idx=int(self.ident.split(":",1)[1]) if ":" in self.ident else int(self.ident)
        except Exception: return False
        # Prefer MSMF on Windows
        backends = []
        try:
            backends.append(getattr(cv2,"CAP_MSMF"))
        except Exception:
            pass
        try:
            backends.append(getattr(cv2,"CAP_DSHOW"))
        except Exception:
            pass
        if not backends:
            backends = [getattr(cv2,'CAP_ANY', 0)]
        cap = None
        for b in backends:
            c = cv2.VideoCapture(idx, b)
            if c is not None and c.isOpened():
                cap = c
                break
            else:
                if c is not None:
                    c.release()
        self.cap = cap
        self.opened = bool(self.cap and self.cap.isOpened())
        return self.opened
    def close(self):
        try:
            if self.cap is not None: self.cap.release()
        finally:
            self.cap=None; self.opened=False; self._streaming=False
    def read(self):
        if not self.opened: return None
        ok, fr = self.cap.read()
        if not ok: return None
        return (time.time(), fr)

class SpinnakerCamera(BaseCamera):
    def __init__(self, ident:str):
        super().__init__(ident); self.cam=None
    def _match_index(self)->Optional[int]:
        sysm=_SpinSystemHolder.get()
        if sysm is None: return None
        cam_list=sysm.GetCameras()
        try:
            n=cam_list.GetSize()
            raw = self.ident.strip()
            prefer_serial=None; prefer_inst=None; prefer_idx=None
            if raw.startswith("serial:"): prefer_serial=raw.split(":",1)[1]
            elif raw.startswith("inst:"): prefer_inst=raw.split(":",1)[1]
            elif raw.startswith("idx:"):
                try: prefer_idx=int(raw.split(":",1)[1])
                except Exception: prefer_idx=None
            elif raw.isdigit():
                prefer_serial=raw  # bare serial support
            # serial
            if prefer_serial:
                for i in range(n):
                    c=cam_list.GetByIndex(i); tl=c.GetTLDeviceNodeMap()
                    ser=_get_str(tl,"DeviceSerialNumber")
                    if ser==prefer_serial: return i
            # instance
            if prefer_inst:
                for i in range(n):
                    c=cam_list.GetByIndex(i); tl=c.GetTLDeviceNodeMap()
                    inst=""
                    for nm in POSSIBLE_INSTANCE_NAMES:
                        inst=_get_str(tl,nm)
                        if inst: break
                    if inst==prefer_inst: return i
            # idx
            if prefer_idx is not None and 0<=prefer_idx<n:
                return prefer_idx
            return None
        finally:
            cam_list.Clear(); del cam_list
    def open(self)->bool:
        if not PYSPIN_OK: return False
        sysm=_SpinSystemHolder.get()
        if sysm is None: return False
        idx=self._match_index()
        if idx is None: idx=0
        cam_list=sysm.GetCameras()
        try:
            if idx>=cam_list.GetSize(): return False
            cam=cam_list.GetByIndex(idx); self.cam=cam
            cam.Init()
            self._configure_nodes(cam)
            self._begin()
            self.opened=True
            return True
        except Exception as e:
            safe_print("[PySpin] open error:", e)
            try:
                if self.cam is not None:
                    self._end(); self.cam.DeInit()
            except Exception: pass
            self.cam=None; self.opened=False
            return False
        finally:
            cam_list.Clear(); del cam_list
    def _configure_nodes(self, cam):
        try:
            nm=cam.GetNodeMap()
            # Acquisition continuous
            try:
                acq=PySpin.CEnumerationPtr(nm.GetNode("AcquisitionMode"))
                if PySpin.IsAvailable(acq) and PySpin.IsWritable(acq):
                    cont=acq.GetEntryByName("Continuous")
                    if PySpin.IsAvailable(cont) and PySpin.IsReadable(cont): acq.SetIntValue(cont.GetValue())
            except Exception: pass
            # Mono8
            try:
                pf=PySpin.CEnumerationPtr(nm.GetNode("PixelFormat"))
                if PySpin.IsAvailable(pf) and PySpin.IsWritable(pf):
                    mono=pf.GetEntryByName("Mono8")
                    if PySpin.IsAvailable(mono) and PySpin.IsReadable(mono): pf.SetIntValue(mono.GetValue())
            except Exception: pass
            # Trigger off
            try:
                tm=PySpin.CEnumerationPtr(nm.GetNode("TriggerMode"))
                if PySpin.IsAvailable(tm) and PySpin.IsWritable(tm):
                    off=tm.GetEntryByName("Off")
                    if PySpin.IsAvailable(off) and PySpin.IsReadable(off): tm.SetIntValue(off.GetValue())
            except Exception: pass
        except Exception: pass
    def _begin(self):
        if self.cam is None: return
        try:
            self.cam.BeginAcquisition(); self._streaming=True
        except Exception as e:
            if "already streaming" in str(e).lower(): self._streaming=True
            else: raise
    def _end(self):
        if self.cam is None: return
        try: self.cam.EndAcquisition()
        except Exception: pass
        self._streaming=False
    def close(self):
        try:
            self._end()
            if self.cam is not None:
                try: self.cam.DeInit()
                except Exception: pass
        finally:
            self.cam=None; self.opened=False; self._streaming=False
    def read(self):
        if not self.opened or self.cam is None: return None
        try:
            img=self.cam.GetNextImage(500)  # 0.5s timeout
            try:
                if img.IsIncomplete(): return None
                arr=img.GetNDArray()
                return (time.time(), arr.copy())
            finally:
                img.Release()
        except Exception as e:
            # Try to restart if needed
            if "not started" in str(e).lower():
                try: self._begin()
                except Exception: pass
            return None

# ----------------- Live Preview (non-modal) -----------------
class LivePreview(QtWidgets.QDialog):
    def __init__(self, cam: BaseCamera, title:str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title); self.cam=cam
        self.lbl=QtWidgets.QLabel(""); self.lbl.setMinimumSize(360,270); self.lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.info=QtWidgets.QLabel("fps: --")
        v=QtWidgets.QVBoxLayout(self); v.addWidget(self.lbl,1); v.addWidget(self.info,0)
        self._timer=QtCore.QTimer(self); self._timer.timeout.connect(self._tick); self._timer.start(1)
        self._frames=0; self._t0=time.time()
    def _tick(self):
        try:
            f=self.cam.read()
            if f is None: return
            _, frame=f; self._frames+=1
            if self._frames%12==0:
                dt=max(1e-3,time.time()-self._t0); self.info.setText(f"fps: {self._frames/dt:.1f}")
            if frame.ndim==2:
                q=QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_Grayscale8)
            else:
                rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                q=QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888)
            self.lbl.setPixmap(QtGui.QPixmap.fromImage(q).scaled(self.lbl.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        except Exception:
            pass  # keep UI alive

# ----------------- Recorder -----------------
class Recorder(QtCore.QObject):
    finished = QtCore.pyqtSignal(dict)  # emits summary dict
    errored  = QtCore.pyqtSignal(str)

    def __init__(self, cams: List[Tuple[int, BaseCamera]], out_root: str, duration_s: float, fourcc: str, fps_hint: float):
        super().__init__()
        self.cams = cams
        self.out_root = ensure_dir(out_root)
        self.duration_s = float(duration_s)
        self.fourcc = fourcc.upper()
        self.fps_hint = float(fps_hint)
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _record_one(self, idx:int, cam: BaseCamera, results:dict):
        try:
            subdir = ensure_dir(os.path.join(self.out_root, ts_now()))
            outfile = os.path.join(subdir, f"cam{idx}_{ts_now()}.mp4" if self.fourcc!="MJPG" else f"cam{idx}_{ts_now()}.avi")
            first = cam.read()
            if first is None:
                raise RuntimeError(f"Cam{idx}: no frame to start recording")
            _, fr0 = first
            h, w = fr0.shape[:2]
            if fr0.ndim==2:
                fr0 = cv2.cvtColor(fr0, cv2.COLOR_GRAY2BGR)
            vw = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*self.fourcc), self.fps_hint, (w,h), isColor=True)
            if not vw.isOpened():
                raise RuntimeError(f"Cam{idx}: failed to open writer")
            t0=time.time(); frames=0
            vw.write(fr0); frames+=1
            while (time.time()-t0) < self.duration_s and not self._stop.is_set():
                f=cam.read()
                if f is None:
                    continue
                _, fr = f
                if fr.ndim==2:
                    fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
                vw.write(fr); frames+=1
            vw.release()
            dt=max(1e-3,time.time()-t0)
            results[idx] = {"path": outfile, "frames": frames, "fps": frames/dt, "w": w, "h": h}
        except Exception as e:
            results[idx] = {"error": str(e)}

    @QtCore.pyqtSlot()
    def run(self):
        # Record each cam in its own thread
        threads=[]; results={}
        for idx, cam in self.cams:
            th=threading.Thread(target=self._record_one, args=(idx,cam,results), daemon=True)
            th.start(); threads.append(th)
        for th in threads:
            th.join()
        # Write metadata
        meta_path=os.path.join(self.out_root, f"record_{ts_now()}.json")
        try:
            with open(meta_path,"w",encoding="utf-8") as f:
                json.dump({"version": APP_VERSION, "results": results, "fourcc": self.fourcc, "fps_hint": self.fps_hint}, f, indent=2)
        except Exception as e:
            safe_print("[Record] metadata write error:", e)
        self.finished.emit(results)

# ----------------- Camera UI Box -----------------
class CameraBox(QtWidgets.QGroupBox):
    wantOpenChanged = QtCore.pyqtSignal(bool)
    def __init__(self, title:str, parent=None):
        super().__init__(title, parent)
        self.ident_edit=QtWidgets.QLineEdit()
        self.ident_edit.setPlaceholderText("Pick from dropdowns below (PySpin preferred for Blackfly)")
        self.bt_open=QtWidgets.QPushButton("Open")
        self.bt_close=QtWidgets.QPushButton("Close")
        self.bt_live=QtWidgets.QPushButton("Live Preview")
        self.bt_live.setEnabled(False)
        self.bt_refresh_local=QtWidgets.QPushButton("Refresh ▼")

        self.combo_spin=QtWidgets.QComboBox(); self.combo_cv=QtWidgets.QComboBox()
        self.combo_spin.setMinimumWidth(480); self.combo_cv.setMinimumWidth(240)
        self.combo_spin.setToolTip("Spinnaker (PySpin) devices — choose by Serial or InstanceID (USB port path)")
        self.combo_cv.setToolTip("UVC/OpenCV devices. Blackfly (Spinnaker) cameras will NOT show here.")
        self.combo_spin.currentIndexChanged.connect(self._spin_changed)
        self.combo_cv.currentIndexChanged.connect(self._cv_changed)
        self.bt_refresh_local.clicked.connect(self._local_refresh)

        form=QtWidgets.QFormLayout()
        form.addRow("Ident:", self.ident_edit)
        row1=QtWidgets.QHBoxLayout(); row1.addWidget(self.combo_spin,1); row1.addWidget(self.bt_refresh_local,0)
        form.addRow("PySpin devices:", row1)
        form.addRow("OpenCV devices:", self.combo_cv)

        btnrow=QtWidgets.QHBoxLayout()
        btnrow.addWidget(self.bt_open); btnrow.addWidget(self.bt_close); btnrow.addWidget(self.bt_live)

        v=QtWidgets.QVBoxLayout(self)
        v.addLayout(form); v.addLayout(btnrow)

        self.cam: Optional[BaseCamera]=None
        self._live_windows: List[LivePreview]=[]

        self.bt_open.clicked.connect(self._open)
        self.bt_close.clicked.connect(self._close)
        self.bt_live.clicked.connect(self._live)

        self._cached_spin=[]; self._cached_cv=[]

    def populate(self, spin_list: List[Dict], cv_list: List[Dict]):
        self._cached_spin=spin_list; self._cached_cv=cv_list
        self.combo_spin.blockSignals(True); self.combo_cv.blockSignals(True)
        self.combo_spin.clear(); self.combo_cv.clear()
        for it in spin_list: self.combo_spin.addItem(it["label"], it["ident"])
        for it in cv_list: self.combo_cv.addItem(it["label"], it["ident"])
        self.combo_spin.blockSignals(False); self.combo_cv.blockSignals(False)
        # Hide/disable OpenCV row if none found
        self.combo_cv.setEnabled(bool(cv_list))
        self.combo_cv.setVisible(bool(cv_list))

    def _local_refresh(self):
        self.populate(self._cached_spin, self._cached_cv)

    def _spin_changed(self, idx):
        ident=self.combo_spin.itemData(idx)
        if ident: self.ident_edit.setText(str(ident))

    def _cv_changed(self, idx):
        ident=self.combo_cv.itemData(idx)
        if ident: self.ident_edit.setText(str(ident))

    def _open(self):
        ident=self.ident_edit.text().strip()
        if not ident:
            QtWidgets.QMessageBox.warning(self,"No ident","Pick a device from a dropdown first."); return
        # close existing
        self._close()
        # choose backend
        cam=None
        if ident.startswith("serial:") or ident.startswith("idx:") or ident.startswith("inst:") or ident.isdigit():
            if PYSPIN_OK: cam=SpinnakerCamera(ident)
            else:
                QtWidgets.QMessageBox.critical(self,"PySpin not available","PySpin not installed/loaded."); return
        elif ident.startswith("index:"):
            cam=OpenCVCamera(ident)
        else:
            QtWidgets.QMessageBox.critical(self,"Unknown ident", f"Don't know how to open: {ident}"); return

        if not cam.open():
            # Helpful hint if user tried a UVC/OpenCV index with a Spinnaker-only camera
            if ident.startswith("index:"):
                QtWidgets.QMessageBox.critical(self, "Open failed",
                    "Failed to open OpenCV device.\n\n"
                    "Note: FLIR/Blackfly cameras use the Spinnaker driver and will not appear as OpenCV 'index:X'.\n"
                    "Please select a device from the 'PySpin devices' dropdown above (by Serial or InstanceID).")
            else:
                QtWidgets.QMessageBox.critical(self, "Open failed", f"Failed to open {ident}")
            return
        cam.start(); self.cam=cam; self.bt_live.setEnabled(True); self.wantOpenChanged.emit(True)

    def _close(self):
        if self.cam is not None:
            try: self.cam.stop(); self.cam.close()
            except Exception: pass
        self.cam=None; self.bt_live.setEnabled(False); self.wantOpenChanged.emit(False)

    def _live(self):
        if self.cam is None: return
        dlg=LivePreview(self.cam, f"Preview — {self.title()}")
        dlg.setModal(False); dlg.resize(680,510); dlg.show()
        self._live_windows.append(dlg)  # keep reference

# ----------------- Main Window -----------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"FlyPy — {APP_VERSION}")
        self.resize(1180, 760)

        # Top bar
        top=QtWidgets.QWidget(); tl=QtWidgets.QHBoxLayout(top)
        self.bt_refresh=QtWidgets.QPushButton("Refresh Cameras")
        self.bt_refresh.setToolTip("Re-enumerate PySpin and OpenCV devices")
        tl.addWidget(self.bt_refresh); tl.addStretch(1)

        # Camera panes
        self.cam0=CameraBox("Camera 0"); self.cam1=CameraBox("Camera 1")

        # Recording controls
        recgrp=QtWidgets.QGroupBox("Recording")
        form=QtWidgets.QGridLayout(recgrp)
        self.spn_duration=QtWidgets.QDoubleSpinBox(); self.spn_duration.setRange(0.1, 600.0); self.spn_duration.setValue(2.0); self.spn_duration.setSuffix(" s")
        self.spn_fps=QtWidgets.QSpinBox(); self.spn_fps.setRange(1, 1000); self.spn_fps.setValue(120)
        self.cmb_codec=QtWidgets.QComboBox(); self.cmb_codec.addItems(["mp4v","MJPG","H264","H265"])
        self.ed_outroot=QtWidgets.QLineEdit(os.path.join(os.getcwd(),"FlyPy_Output"))
        self.bt_browse=QtWidgets.QPushButton("Browse…")
        self.bt_record_both=QtWidgets.QPushButton("Record BOTH")
        self.bt_record_0=QtWidgets.QPushButton("Record Cam0")
        self.bt_record_1=QtWidgets.QPushButton("Record Cam1")
        form.addWidget(QtWidgets.QLabel("Duration:"), 0,0); form.addWidget(self.spn_duration,0,1)
        form.addWidget(QtWidgets.QLabel("FPS hint:"), 0,2); form.addWidget(self.spn_fps,0,3)
        form.addWidget(QtWidgets.QLabel("Codec:"), 1,0); form.addWidget(self.cmb_codec,1,1)
        form.addWidget(QtWidgets.QLabel("Output Root:"), 1,2); 
        outrow=QtWidgets.QHBoxLayout(); outrow.addWidget(self.ed_outroot,1); outrow.addWidget(self.bt_browse,0)
        form.addLayout(outrow, 1,3)
        form.addWidget(self.bt_record_both, 2,0,1,2); form.addWidget(self.bt_record_0,2,2); form.addWidget(self.bt_record_1,2,3)

        # Log/status
        self.log=QtWidgets.QTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(140)

        central=QtWidgets.QWidget(); cl=QtWidgets.QVBoxLayout(central)
        cl.addWidget(top,0)
        hh=QtWidgets.QHBoxLayout(); hh.addWidget(self.cam0,1); hh.addWidget(self.cam1,1)
        cl.addLayout(hh,1)
        cl.addWidget(recgrp,0)
        cl.addWidget(self.log,1)
        self.setCentralWidget(central)

        self.status=self.statusBar()

        # Signals
        self.bt_refresh.clicked.connect(self.refresh_cameras)
        self.bt_browse.clicked.connect(self._browse_out)
        self.bt_record_both.clicked.connect(self._rec_both)
        self.bt_record_0.clicked.connect(lambda: self._rec_only(0))
        self.bt_record_1.clicked.connect(lambda: self._rec_only(1))

        # first inventory
        self.refresh_cameras()

        # recorder thread bits
        self._rec_thread=None; self._rec_worker=None

    def _browse_out(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output root", self.ed_outroot.text())
        if d:
            self.ed_outroot.setText(d)

    def logmsg(self, s:str):
        safe_print(s)
        self.log.append(s)

    def _toggle_rec_buttons(self, en:bool):
        for b in (self.bt_record_both, self.bt_record_0, self.bt_record_1): b.setEnabled(en)

    def _collect_cams(self):
        cams=[]
        if self.cam0.cam is not None: cams.append((0,self.cam0.cam))
        if self.cam1.cam is not None: cams.append((1,self.cam1.cam))
        return cams

    def _start_recorder(self, which:Optional[int]):
        cams = self._collect_cams()
        if which is not None:
            cams = [c for c in cams if c[0]==which]
        if not cams:
            QtWidgets.QMessageBox.warning(self,"No camera","Open a camera first."); return
        outroot=self.ed_outroot.text().strip() or os.path.join(os.getcwd(),"FlyPy_Output")
        dur=self.spn_duration.value(); fps=self.spn_fps.value(); fourcc=self.cmb_codec.currentText()
        self._rec_worker=Recorder(cams, outroot, dur, fourcc, fps)
        self._rec_thread=QtCore.QThread(self); self._rec_worker.moveToThread(self._rec_thread)
        self._rec_thread.started.connect(self._rec_worker.run)
        self._rec_worker.finished.connect(self._on_rec_done)
        self._rec_worker.errored.connect(self._on_rec_err)
        self._rec_thread.start()
        self._toggle_rec_buttons(False)
        self.logmsg(f"[Record] Started: {len(cams)} cam(s), {dur}s, fourcc={fourcc}, fps_hint={fps}")

    def _rec_both(self):
        self._start_recorder(which=None)

    def _rec_only(self, idx:int):
        self._start_recorder(which=idx)

    @QtCore.pyqtSlot(dict)
    def _on_rec_done(self, results:dict):
        self._toggle_rec_buttons(True)
        try:
            self._rec_thread.quit(); self._rec_thread.wait()
        except Exception: pass
        self._rec_thread=None; self._rec_worker=None
        ok_list=[]; err_list=[]
        for k,v in results.items():
            if isinstance(v, dict) and "error" in v: err_list.append((k,v["error"]))
            else: ok_list.append((k,v))
        for k,v in ok_list:
            self.logmsg(f"[Record] cam{k}: saved {os.path.basename(v['path'])} ({v['frames']} frames @ {v['fps']:.1f} fps)")
        for k,msg in err_list:
            self.logmsg(f"[Record] cam{k}: ERROR: {msg}")
        self.status.showMessage("Recording finished", 4000)

    @QtCore.pyqtSlot(str)
    def _on_rec_err(self, msg:str):
        self._toggle_rec_buttons(True)
        try:
            self._rec_thread.quit(); self._rec_thread.wait()
        except Exception: pass
        self._rec_thread=None; self._rec_worker=None
        QtWidgets.QMessageBox.critical(self,"Recording error", msg)
        self.logmsg(f"[Record] ERROR: {msg}")

    def refresh_cameras(self):
        self.status.showMessage("Enumerating cameras…")
        try: spin=enum_pyspin()
        except Exception as e: safe_print("[UI] enum_pyspin error:", e); spin=[]
        try: cv=enum_opencv()
        except Exception as e: safe_print("[UI] enum_opencv error:", e); cv=[]
        self.cam0.populate(spin, cv); self.cam1.populate(spin, cv)

        # Default idents to first two PySpin entries to avoid 'index:0' confusion
        if spin:
            self.cam0.ident_edit.setText(spin[0]["ident"])
            if len(spin) >= 2:
                self.cam1.ident_edit.setText(spin[1]["ident"])
            else:
                self.cam1.ident_edit.setPlaceholderText("Connect another camera or click Refresh Cameras")
        else:
            self.cam0.ident_edit.clear()
            self.cam1.ident_edit.clear()

        if self.cam0.ident_edit.text().strip() == self.cam1.ident_edit.text().strip() and self.cam0.ident_edit.text().strip():
            self.status.showMessage("Warning: both camera idents are identical. Select different PySpin entries.", 6000)
        else:
            self.status.showMessage(f"Found PySpin:{len(spin)} OpenCV:{len(cv)}", 3000)

    def closeEvent(self, ev: QtGui.QCloseEvent):
        try: self.cam0._close(); self.cam1._close()
        finally:
            try: _SpinSystemHolder.release()
            except Exception: pass
        return super().closeEvent(ev)

def main():
    print("=== FlyPy Startup ===")
    print(f"Version: {APP_VERSION}")
    print(f"OpenCV: {'OK' if OPENCV_OK else 'not available'}")
    print(f"PySpin: {'OK' if PYSPIN_OK else 'not available'}")
    print("PsychoPy: not available (this GUI focuses on cameras/recording)")
    print("======================")
    app=QtWidgets.QApplication(sys.argv)
    win=MainWindow(); win.show()
    return app.exec_()

if __name__ == "__main__":
    try: sys.exit(main())
    except Exception as e:
        traceback.print_exc(); sys.exit(1)
