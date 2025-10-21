
# FlyPy — Unified Trigger → Cameras + Lights + Looming Stimulus
# v1.12.0
# - NEW: "Refresh Cameras" button that scans PySpin + OpenCV devices and populates per-camera dropdowns.
# - NEW: Per-camera "Live Preview (high-FPS)" window driven by a background thread to keep GUI responsive.
# - FIX: PySpin camera selection now prefers DeviceSerialNumber and supports "idx=N" or "index:N".
# - FIX: Avoid opening same physical PySpin device twice by matching actual serials; clearer logs.
# - TWEAK: ROI writes only when nodes are writable; fewer noisy AccessException logs.
#
# Notes:
# * PsychoPy is optional. If import fails, OpenCV fallback is used for the looming stimulus.
# * Blackfly presets target Mono8 @ ~522 fps with 640x512 ROI. Adjust exposure to meet frame period.
#
# SPDX-License-Identifier: MIT

import os, sys, time, csv, atexit, threading, queue, re
from collections import deque
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
import importlib, numpy as np

from PyQt5 import QtWidgets, QtCore, QtGui

__version__ = "1.12.0"

# HiDPI
try:
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
except Exception:
    pass

# stdout
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Optional libs
try:
    import cv2; HAVE_OPENCV=True
except Exception:
    HAVE_OPENCV=False; cv2=None  # type: ignore

# PsychoPy lazy
PSY_LOADED=None; visual=None; core=None
def _ensure_psychopy_loaded()->bool:
    global PSY_LOADED, visual, core
    if PSY_LOADED is True: return True
    if PSY_LOADED is False: return False
    try:
        importlib.import_module("psychopy")
        visual=importlib.import_module("psychopy.visual")
        core  =importlib.import_module("psychopy.core")
        PSY_LOADED=True; return True
    except Exception:
        visual=None; core=None; PSY_LOADED=False; return False

def ensure_dir(p:str): os.makedirs(p, exist_ok=True)
def now_stamp()->str: return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
def day_folder(root:str)->str:
    d=datetime.now().strftime("%Y%m%d"); path=os.path.join(root,d); ensure_dir(path); return path
def wait_s(sec:float):
    if _ensure_psychopy_loaded():
        try: core.wait(sec); return
        except Exception: pass
    time.sleep(sec)

VIDEO_PRESETS=[
    {"id":"mp4_mp4v","label":"MP4 / mp4v — very compatible; moderate CPU","fourcc":"mp4v"},
    {"id":"avi_xvid","label":"AVI / XVID — broad compatibility; larger files","fourcc":"XVID"},
    {"id":"avi_mjpg","label":"AVI / MJPG — very large files; light CPU (fast write)","fourcc":"MJPG"},
]
PRESETS_BY_ID={p["id"]:p for p in VIDEO_PRESETS}
def default_preset_id()->str: return "avi_mjpg"  # prefer MJPG for high-FPS

# -------------------- Config --------------------
class Config:
    def __init__(self):
        self.simulation_mode=False
        self.sim_trigger_interval=5.0
        self.output_root="FlyPy_Output"
        self.prewarm_stim=False

        self.video_preset_id=default_preset_id()
        self.fourcc=PRESETS_BY_ID[self.video_preset_id]["fourcc"]
        self.record_duration_s=3.0

        self.stim_duration_s=1.5
        self.stim_r0_px=8
        self.stim_r1_px=240
        self.stim_bg_grey=1.0
        self.lights_delay_s=0.0
        self.stim_delay_s=0.0
        self.stim_screen_index=0
        self.stim_fullscreen=False
        self.gui_screen_index=0

        # Camera defaults (preset aims for Blackfly 522 fps)
        self.cam0_backend="PySpin"; self.cam1_backend="PySpin"
        self.cam0_id=""; self.cam1_id=""
        self.cam0_target_fps=522; self.cam1_target_fps=522
        self.cam0_width=640; self.cam0_height=512
        self.cam1_width=640; self.cam1_height=512
        self.cam0_exposure_us=1500; self.cam1_exposure_us=1500  # µs
        self.cam0_hw_trigger=False; self.cam1_hw_trigger=False
        self.cam_async_writer=True

# -------------------- Hardware Bridge --------------------
class HardwareBridge:
    def __init__(self,cfg:Config,port:str=None,baud:int=115200):
        self.cfg=cfg; self.simulated=cfg.simulation_mode; self.port=port; self.baud=baud
        self._opened=False; self._last_sim=time.time(); self.ser=None
    def _autodetect_port(self)->Optional[str]:
        try:
            import serial.tools.list_ports
            for p in serial.tools.list_ports.comports():
                vid=f"{p.vid:04X}" if p.vid is not None else None
                pid=f"{p.pid:04X}" if p.pid is not None else None
                if vid=="1A86" and pid=="7523": return p.device
            for p in serial.tools.list_ports.comports():
                d=(p.description or "").lower()
                if "ch340" in d or "uno" in d or "elegoo" in d: return p.device
        except Exception: pass
        return None
    def _open_if_needed(self):
        if self.simulated or self._opened: return
        self._opened=True
        try:
            import serial
            if not self.port: self.port=self._autodetect_port()
            if self.port:
                try:
                    self.ser=serial.Serial(self.port,self.baud,timeout=0.01); wait_s(1.2); print(f"[HW] Serial open: {self.port} @ {self.baud}")
                except Exception as e:
                    print(f"[HW] Open failed: {e} → simulation"); self.simulated=True
            else:
                print("[HW] No CH340/UNO port found → simulation"); self.simulated=True
        except Exception:
            print("[HW] pyserial not available → simulation"); self.simulated=True
    def check_trigger(self)->bool:
        if self.simulated:
            now=time.time()
            if now-self._last_sim>=self.cfg.sim_trigger_interval:
                self._last_sim=now; print("[HW] (Sim) Trigger"); return True
            return False
        self._open_if_needed()
        try:
            if self.ser and self.ser.in_waiting:
                line=self.ser.readline().decode(errors="ignore").strip()
                if line=="T": return True
        except Exception as e: print(f"[HW] Read error: {e}")
        return False
    def _send(self,text:str):
        self._open_if_needed()
        if self.simulated or not self.ser: print(f"[HW] (Sim) SEND: {text}"); return
        try: self.ser.write((text.strip()+"\n").encode("utf-8",errors="ignore"))
        except Exception as e: print(f"[HW] Write error: {e}")
    def mark_start(self): self._send("MARK START")
    def mark_end(self):   self._send("MARK END")
    def lights_on(self):  self._send("LIGHT ON")
    def lights_off(self): self._send("LIGHT OFF")
    def close(self):
        if not self.simulated and self.ser:
            try: self.ser.close()
            except Exception: pass
        self.ser=None; self._opened=False

# -------------------- Camera backends --------------------
class BaseCamera:
    def open(self): raise NotImplementedError
    def get_frame(self): raise NotImplementedError  # returns BGR ndarray (uint8)
    def release(self): raise NotImplementedError
    def frame_size(self)->Tuple[int,int]: raise NotImplementedError
    def start_acquisition(self): pass
    def stop_acquisition(self): pass

# ---- OpenCV backend (simple; for testing) ----
class OpenCVCamera(BaseCamera):
    def __init__(self,index:int,target_fps:float):
        if not HAVE_OPENCV: raise RuntimeError("OpenCV is not installed")
        self.index=index; self.target_fps=float(target_fps); self.cap=None
    def open(self):
        backends=[cv2.CAP_ANY]; 
        if os.name=="nt": backends=[cv2.CAP_DSHOW,cv2.CAP_MSMF,cv2.CAP_ANY]
        for be in backends:
            try:
                cap=cv2.VideoCapture(self.index,be)
                if cap and cap.isOpened():
                    try:
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y800"))  # gray if supported
                        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                    except Exception: pass
                    try: cap.set(cv2.CAP_PROP_FPS,float(self.target_fps))
                    except Exception: pass
                    self.cap=cap; return
                if cap: cap.release()
            except Exception: pass
        self.cap=None
    def start_acquisition(self): pass
    def get_frame(self):
        if not self.cap: return None
        ok,frame=self.cap.read()
        if not ok or frame is None: return None
        if frame.ndim==2:
            frame=cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame
    def frame_size(self): 
        if self.cap: return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480))
        return (640,480)
    def stop_acquisition(self): pass
    def release(self):
        if self.cap:
            try: self.cap.release()
            except Exception: pass
            self.cap=None

# ---- Spinnaker (PySpin) backend ----
HAVE_PYSPIN=False; PySpin=None; _SPIN_SYS=None; _SPIN_REF=0

def _spin_system_get():
    global PySpin, HAVE_PYSPIN, _SPIN_SYS, _SPIN_REF
    if not HAVE_PYSPIN:
        try:
            import PySpin as _ps  # type: ignore
            PySpin=_ps; HAVE_PYSPIN=True
        except Exception as e:
            print(f"[PySpin] Not available: {e}")
            return None
    if _SPIN_SYS is None:
        _SPIN_SYS = PySpin.System.GetInstance()
    _SPIN_REF += 1
    return _SPIN_SYS

def _spin_system_release():
    global _SPIN_SYS, _SPIN_REF
    _SPIN_REF = max(0, _SPIN_REF-1)
    if _SPIN_REF==0 and _SPIN_SYS is not None:
        try: _SPIN_SYS.ReleaseInstance()
        except Exception: pass
        _SPIN_SYS=None

def _safe_set_enum(nodemap, name, symbolic):
    try:
        node = PySpin.CEnumerationPtr(nodemap.GetNode(name))
        if not PySpin.IsWritable(node): return False
        entry = node.GetEntryByName(symbolic)
        if not PySpin.IsReadable(entry): return False
        node.SetIntValue(entry.GetValue())
        return True
    except Exception as e:
        print(f"[PySpin] Enum {name}={symbolic} failed: {e}")
        return False

def _safe_set_float(nodemap, name, value):
    try:
        node = PySpin.CFloatPtr(nodemap.GetNode(name))
        if not PySpin.IsWritable(node): return False
        lo = node.GetMin(); hi = node.GetMax()
        v = max(lo, min(hi, float(value)))
        node.SetValue(v)
        return True
    except Exception as e:
        print(f"[PySpin] Float {name}={value} failed: {e}")
        return False

def _safe_set_bool(nodemap, name, value:bool):
    try:
        node = PySpin.CBooleanPtr(nodemap.GetNode(name))
        if not PySpin.IsWritable(node): return False
        node.SetValue(bool(value)); return True
    except Exception as e:
        print(f"[PySpin] Bool {name}={value} failed: {e}")
        return False

def _align_to_inc(val, inc, lo, hi):
    if inc <= 0: return int(max(lo, min(hi, val)))
    v = int(val // inc * inc)
    return max(int(lo), min(int(hi), int(v)))

def _parse_ident_to_selector(ident:str)->Dict[str,Any]:
    """
    Return dict like {'mode': 'serial', 'value': '24102017'} or {'mode':'index','value':1}
    Accepted forms: '24102017', 'idx=1', 'idx:1', 'index=1', 'index:1'
    """
    s=(ident or "").strip()
    if not s:
        return {'mode':'first','value':0}
    m=re.match(r'^(?:idx|index)[:=]\s*(\d+)$', s, re.I)
    if m:
        return {'mode':'index','value':int(m.group(1))}
    if s.isdigit() and len(s)>=5:
        return {'mode':'serial','value':s}
    # fallback: try exact UniqueID too
    return {'mode':'unique','value':s}

def _spin_list_devices()->List[Dict[str,Any]]:
    """List PySpin cameras without opening them. Returns list of dicts with keys: index, serial, model, unique"""
    res=[]
    sys_inst=_spin_system_get()
    if sys_inst is None: return res
    lst=sys_inst.GetCameras()
    try:
        n=lst.GetSize()
        for i in range(n):
            cam=lst.GetByIndex(i)
            serial=""; model=""; unique=""
            try:
                tl=cam.GetTLDeviceNodeMap()
                s_node=PySpin.CStringPtr(tl.GetNode("DeviceSerialNumber"))
                if s_node and PySpin.IsReadable(s_node): serial=s_node.GetValue()
                m_node=PySpin.CStringPtr(tl.GetNode("DeviceModelName"))
                if m_node and PySpin.IsReadable(m_node): model=m_node.GetValue()
                u_node=PySpin.CStringPtr(tl.GetNode("DeviceID"))
                if u_node and PySpin.IsReadable(u_node): unique=u_node.GetValue()
            except Exception:
                try: unique=cam.GetUniqueID()
                except Exception: unique=""
            res.append({"index":i,"serial":serial,"model":model,"unique":unique})
    except Exception as e:
        print(f"[PySpin] List error: {e}")
    finally:
        try: lst.Clear()
        except Exception: pass
        _spin_system_release()
    return res

class SpinnakerCamera(BaseCamera):
    def __init__(self, ident:str, target_fps:float, width:int=0, height:int=0, exposure_us:int=1500, hw_trigger:bool=False):
        self.ident = (ident or "").strip()
        self.sel = _parse_ident_to_selector(self.ident)
        self.target_fps = float(target_fps)
        self.req_w = int(width); self.req_h = int(height)
        self.exposure_us = int(exposure_us)
        self.hw_trigger = bool(hw_trigger)
        self.cam = None; self.node = None; self.snode = None; self.anode = None
        self._acq = False; self._last_size = (640,480); self._mono = True
        self._reported_serial=""; self._reported_model=""; self._reported_index=None

    def open(self):
        sys_inst = _spin_system_get()
        if sys_inst is None:
            raise RuntimeError("PySpin not available; install Spinnaker SDK + PySpin and ensure DLLs are in PATH.")
        lst = sys_inst.GetCameras()
        try:
            if lst.GetSize()==0:
                raise RuntimeError("No Spinnaker cameras detected")
            # choose camera
            chosen=None; chosen_idx=0
            # Build descriptor for each
            desc=[]
            for i in range(lst.GetSize()):
                cam=lst.GetByIndex(i)
                serial=""; model=""; unique=""
                try:
                    tl=cam.GetTLDeviceNodeMap()
                    s_node=PySpin.CStringPtr(tl.GetNode("DeviceSerialNumber"))
                    if s_node and PySpin.IsReadable(s_node): serial=s_node.GetValue()
                    m_node=PySpin.CStringPtr(tl.GetNode("DeviceModelName"))
                    if m_node and PySpin.IsReadable(m_node): model=m_node.GetValue()
                    u_node=PySpin.CStringPtr(tl.GetNode("DeviceID"))
                    if u_node and PySpin.IsReadable(u_node): unique=u_node.GetValue()
                except Exception:
                    try: unique=cam.GetUniqueID()
                    except Exception: unique=""
                desc.append((i,cam,serial,model,unique))
            # Selection logic
            mode=self.sel['mode']; val=self.sel['value']
            if mode=='index':
                for (i,cam,serial,model,unique) in desc:
                    if i==val: chosen=cam; chosen_idx=i; self._reported_serial=serial; self._reported_model=model; break
            elif mode=='serial':
                for (i,cam,serial,model,unique) in desc:
                    if serial==str(val): chosen=cam; chosen_idx=i; self._reported_serial=serial; self._reported_model=model; break
            elif mode=='unique':
                for (i,cam,serial,model,unique) in desc:
                    if unique==str(val): chosen=cam; chosen_idx=i; self._reported_serial=serial; self._reported_model=model; break
            if chosen is None:
                chosen=desc[0][1]; chosen_idx=desc[0][0]; self._reported_serial=desc[0][2]; self._reported_model=desc[0][3]
                if self.ident:
                    print(f"[PySpin] Identifier '{self.ident}' not found; using first camera.")
            self.cam = chosen; self._reported_index=chosen_idx
            self.cam.Init()
            self.node = self.cam.GetNodeMap(); self.snode = self.cam.GetTLStreamNodeMap(); self.anode = self.cam.GetTLDeviceNodeMap()

            # Stream config
            _safe_set_enum(self.snode, "StreamBufferHandlingMode", "NewestOnly")
            try:
                mode = PySpin.CEnumerationPtr(self.snode.GetNode("StreamBufferCountMode"))
                if PySpin.IsWritable(mode):
                    mode.SetIntValue(mode.GetEntryByName("Manual").GetValue())
                    cnt = PySpin.CIntegerPtr(self.snode.GetNode("StreamBufferCountManual"))
                    if PySpin.IsWritable(cnt): cnt.SetValue(max(int(cnt.GetMin()), min(int(cnt.GetMax()), 64)))
            except Exception as e:
                print(f"[PySpin] Stream buffer config note: {e}")

            # Pixel format
            ok = _safe_set_enum(self.node, "PixelFormat", "Mono8")
            self._mono = ok
            if not ok:
                ok = _safe_set_enum(self.node, "PixelFormat", "BayerRG8")
                self._mono = False if ok else True

            # ROI
            try:
                w = PySpin.CIntegerPtr(self.node.GetNode("Width"))
                h = PySpin.CIntegerPtr(self.node.GetNode("Height"))
                ox = PySpin.CIntegerPtr(self.node.GetNode("OffsetX"))
                oy = PySpin.CIntegerPtr(self.node.GetNode("OffsetY"))
                if w and h and (PySpin.IsWritable(w) and PySpin.IsWritable(h)):
                    maxw = int(w.GetMax()); maxh = int(h.GetMax())
                    incw = int(w.GetInc()) or 2; inch = int(h.GetInc()) or 2
                    reqw = maxw if self.req_w<=0 or self.req_w>maxw else _align_to_inc(self.req_w, incw, w.GetMin(), maxw)
                    reqh = maxh if self.req_h<=0 or self.req_h>maxh else _align_to_inc(self.req_h, inch, h.GetMin(), maxh)
                    cx = max(0, (maxw - reqw)//(2*incw)*incw); cy = max(0, (maxh - reqh)//(2*inch)*inch)
                    if ox and PySpin.IsWritable(ox): ox.SetValue(cx)
                    if oy and PySpin.IsWritable(oy): oy.SetValue(cy)
                    w.SetValue(reqw); h.SetValue(reqh)
                    self._last_size = (int(reqw), int(reqh))
                else:
                    # readonly; keep default
                    pass
            except Exception as e:
                print(f"[PySpin] ROI note: {e}")

            # Exposure/Gain
            _safe_set_enum(self.node, "ExposureAuto", "Off")
            if self.exposure_us>0:
                period_us = 1e6/max(1.0, self.target_fps)
                exp_us = min(self.exposure_us, int(period_us*0.85))
                _safe_set_float(self.node, "ExposureTime", exp_us)
            _safe_set_enum(self.node, "GainAuto", "Off")

            # FPS / Trigger
            if not self.hw_trigger:
                _safe_set_bool(self.node, "AcquisitionFrameRateEnable", True)
                _safe_set_float(self.node, "AcquisitionFrameRate", self.target_fps)
            else:
                _safe_set_bool(self.node, "AcquisitionFrameRateEnable", False)

            if self.hw_trigger:
                _safe_set_enum(self.node, "TriggerMode", "Off")
                _safe_set_enum(self.node, "TriggerSelector", "FrameStart")
                _safe_set_enum(self.node, "TriggerSource", "Line0")
                _safe_set_enum(self.node, "TriggerActivation", "RisingEdge")
                _safe_set_enum(self.node, "TriggerOverlap", "ReadOut")
                _safe_set_enum(self.node, "TriggerMode", "On")
            else:
                _safe_set_enum(self.node, "TriggerMode", "Off")

            _safe_set_enum(self.node, "AcquisitionMode", "Continuous")

            # Throughput
            try:
                dl = PySpin.CFloatPtr(self.node.GetNode("DeviceLinkThroughputLimit"))
                if dl and PySpin.IsWritable(dl):
                    desired_Bps = self._last_size[0]*self._last_size[1]*(1 if self._mono else 1.5)*self.target_fps
                    dl.SetValue(min(dl.GetMax(), max(dl.GetMin(), desired_Bps*1.2)))
            except Exception:
                pass

        except Exception:
            try: lst.Clear()
            except Exception: pass
            _spin_system_release()
            raise
        finally:
            try: lst.Clear()
            except Exception: pass

    def start_acquisition(self):
        if self.cam and not self._acq:
            try: self.cam.BeginAcquisition(); self._acq=True
            except Exception as e: print(f"[PySpin] BeginAcquisition: {e}")

    def get_frame(self):
        if not self.cam: return None
        if not self._acq: self.start_acquisition()
        try:
            img = self.cam.GetNextImage(50)
            if img.IsIncomplete(): img.Release(); return None
            try: arr = img.GetNDArray()
            except Exception:
                try:
                    fmt = "Mono8" if self._mono else "BayerRG8"
                    if fmt != "Mono8":
                        conv = PySpin.ImageProcessor(); conv.SetColorProcessing(PySpin.HQ_LINEAR)
                        img2 = conv.Convert(img, PySpin.PixelFormat_BGR8); arr = img2.GetNDArray()
                    else:
                        arr = img.GetNDArray()
                except Exception:
                    img.Release(); return None
            w = img.GetWidth(); h = img.GetHeight(); img.Release()
            if arr.ndim==2:
                if HAVE_OPENCV: arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                else: arr = np.repeat(arr[...,None],3,axis=2)
            elif arr.shape[2]==1:
                arr = np.repeat(arr,3,axis=2)
            self._last_size=(int(w),int(h))
            return arr
        except Exception as e:
            print(f"[PySpin] Frame error: {e}"); return None

    def frame_size(self): return self._last_size
    def stop_acquisition(self):
        if self.cam and self._acq:
            try: self.cam.EndAcquisition()
            except Exception: pass
            self._acq=False
    def release(self):
        try:
            self.stop_acquisition()
            if self.cam:
                try: self.cam.DeInit()
                except Exception: pass
                try: del self.cam
                except Exception: pass
            self.cam=None
        finally:
            _spin_system_release()

# ---- Camera wrapper ----
class CameraNode:
    def __init__(self, name:str, backend:str, ident:str, target_fps:int, adv=None):
        self.name=name; self.backend=backend; self.ident=ident; self.target_fps=float(target_fps)
        self.dev:Optional[BaseCamera]=None; self.synthetic=False; self.preview_times=deque(maxlen=120); self.adv=adv or {}
        self._live_thread=None; self._live_stop=None; self._live_window=f"Live Preview — {name}"
        self._last_reported_serial=""
    def _open_if_needed(self):
        if self.dev is not None or self.synthetic: return
        try:
            if self.backend=="PySpin":
                ident=self.ident; width=int(self.adv.get("width",0) or 0); height=int(self.adv.get("height",0) or 0)
                exposure_us=int(self.adv.get("exposure_us",1500) or 1500); hw_trig=bool(self.adv.get("hw_trigger",False))
                dev=SpinnakerCamera(ident,self.target_fps,width,height,exposure_us,hw_trig); dev.open()
                self.dev=dev
                opened_id = dev._reported_serial or ident or "(first)"
                self._last_reported_serial=dev._reported_serial or ""
                print(f"[{self.name}] PySpin open: {opened_id} {self.dev.frame_size()} @ {self.target_fps}fps trig={'HW' if hw_trig else 'Free'}")
            else:
                if not HAVE_OPENCV: raise RuntimeError("OpenCV not installed")
                try: idx=int(self.ident or "0")
                except Exception: idx=0
                dev=OpenCVCamera(idx,self.target_fps); dev.open()
                if dev.cap is None: self.synthetic=True; self.dev=None; print(f"[{self.name}] OpenCV index {idx} not available → synthetic")
                else: self.dev=dev; print(f"[{self.name}] OpenCV open: index {idx}")
        except Exception as e:
            print(f"[{self.name}] Open error: {e} → synthetic"); self.dev=None; self.synthetic=True
    def set_backend_ident(self, backend:str, ident:str, adv=None):
        self.release(); self.backend=backend; self.ident=ident; self.synthetic=False
        if adv is not None: self.adv=adv
        print(f"[{self.name}] set backend={backend} ident={ident} (lazy open)")
    def set_target_fps(self,fps:int): self.target_fps=float(fps)
    def grab_preview(self,w:int,h:int):
        self._open_if_needed()
        if self.synthetic or self.dev is None:
            frame=np.full((max(h,1),max(w,1),3),240,dtype=np.uint8)
            if HAVE_OPENCV: cv2.putText(frame,f"{self.name} (synthetic)",(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2,cv2.LINE_AA)
            self.preview_times.append(time.time()); return frame[...,::-1]  # to RGB for Qt
        img=self.dev.get_frame()
        if img is None:
            frame=np.full((max(h,1),max(w,1),3),255,dtype=np.uint8)
            if HAVE_OPENCV: cv2.putText(frame,f"{self.name} [drop]",(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2,cv2.LINE_AA)
            self.preview_times.append(time.time()); return frame[...,::-1]
        if HAVE_OPENCV and (img.shape[1]!=w or img.shape[0]!=h):
            img=cv2.resize(img,(w,h))
        self.preview_times.append(time.time()); return img[...,::-1]  # BGR->RGB
    def driver_fps(self)->float:
        if len(self.preview_times)<2: return 0.0
        dt=self.preview_times[-1]-self.preview_times[0]; n=len(self.preview_times)-1
        return (n/dt) if dt>0 else 0.0
    def probe_max_fps(self, seconds:float=3.0)->Tuple[float,int,int]:
        """Measure achievable FPS without encoding. Returns (fps, frames, drops)."""
        self._open_if_needed()
        frames=0; drops=0
        t0=time.time()
        try: 
            if self.dev: self.dev.start_acquisition()
        except Exception: pass
        while (time.time()-t0)<seconds:
            img=None if (self.synthetic or self.dev is None) else self.dev.get_frame()
            if img is None: drops+=1
            else: frames+=1
        try: 
            if self.dev: self.dev.stop_acquisition()
        except Exception: pass
        elapsed=max(1e-6, time.time()-t0)
        return (frames/elapsed, frames, drops)
    def start_live_preview(self):
        if not HAVE_OPENCV: 
            print("[Live] OpenCV required for live preview"); return
        if self._live_thread and self._live_thread.is_alive():
            return
        self._open_if_needed()
        self._live_stop=threading.Event()
        def run():
            cv2.namedWindow(self._live_window, cv2.WINDOW_NORMAL)
            last=None
            while self._live_stop and not self._live_stop.is_set():
                img=None if (self.synthetic or self.dev is None) else self.dev.get_frame()
                if img is not None:
                    cv2.imshow(self._live_window, img)
                    key=cv2.waitKey(1) & 0xFF
                    if key==27: break
                else:
                    # brief sleep to avoid busy loop
                    if last is None or (time.time()-last)>0.1:
                        last=time.time()
                        blank=np.full((360,480,3),200,dtype=np.uint8)
                        cv2.putText(blank,"[no frame]",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
                        cv2.imshow(self._live_window, blank)
                        cv2.waitKey(1)
                    time.sleep(0.002)
            cv2.destroyWindow(self._live_window)
        self._live_thread=threading.Thread(target=run,daemon=True)
        self._live_thread.start()
        print(f"[Live] Started {self._live_window}")
    def stop_live_preview(self):
        if self._live_stop: self._live_stop.set()
        if self._live_thread:
            self._live_thread.join(timeout=1.0)
        self._live_thread=None; self._live_stop=None
        print(f"[Live] Stopped {self._live_window}")
    def release(self):
        try:
            self.stop_live_preview()
        except Exception: pass
        try:
            if self.dev: self.dev.release()
        except Exception: pass
        self.dev=None; self.synthetic=False

# -------------------- Stimulus --------------------
class LoomingStim:
    def __init__(self,cfg:Config):
        self.cfg=cfg; self._pp_win=None; self._pp_cfg=None; self._cv_window_name="Looming Stimulus"; self._cv_open=False; self._cv_size=(800,600)
    def _pp_window(self,screen_idx:int,fullscreen:bool,bg_grey:float):
        need_new=False
        if self._pp_win is None: need_new=True
        elif self._pp_cfg!=(screen_idx,fullscreen):
            try: self._pp_win.close()
            except Exception: pass
            self._pp_win=None; need_new=True
        if need_new:
            try:
                if fullscreen: self._pp_win=visual.Window(color=[bg_grey]*3,units='pix',fullscr=True,screen=screen_idx)
                else: self._pp_win=visual.Window(size=self._cv_size,color=[bg_grey]*3,units='pix',fullscr=False,screen=screen_idx,allowGUI=True)
                self._pp_cfg=(screen_idx,fullscreen)
            except Exception as e: print(f"[Stim] PsychoPy window error: {e}"); self._pp_win=None
        if self._pp_win is not None:
            try: self._pp_win.color=[bg_grey]*3
            except Exception: pass
    def _cv_window(self,screen_idx:int,bg_grey:float):
        try:
            if not self._cv_open:
                cv2.namedWindow(self._cv_window_name,cv2.WINDOW_NORMAL); cv2.resizeWindow(self._cv_window_name,self._cv_size[0],self._cv_size[1]); self._cv_open=True
            geoms=QtGui.QGuiApplication.screens()
            if 0<=screen_idx<len(geoms):
                g=geoms[screen_idx].geometry(); cv2.moveWindow(self._cv_window_name,g.x()+50,g.y()+50)
            bg=int(max(0,min(255,int(bg_grey*255)))); frame=np.full((self._cv_size[1],self._cv_size[0],3),bg,dtype=np.uint8)
            cv2.imshow(self._cv_window_name,frame); cv2.waitKey(1)
        except Exception as e: print(f"[Stim] OpenCV window error: {e}"); self._cv_open=False
    def open_persistent(self,screen_idx:int,fullscreen:bool,bg_grey:float):
        if _ensure_psychopy_loaded():
            self._pp_window(screen_idx,fullscreen,bg_grey)
            if self._pp_win is not None:
                try: self._pp_win.flip()
                except Exception: pass
        else: self._cv_window(screen_idx,bg_grey)
    def run(self,duration_s:float,r0:int,r1:int,bg_grey:float,screen_idx:int,fullscreen:bool):
        print("[Stim] Looming start")
        if _ensure_psychopy_loaded():
            try:
                self._pp_window(screen_idx,fullscreen,bg_grey)
                if self._pp_win is not None:
                    dot=visual.Circle(self._pp_win,radius=r0,fillColor='black',lineColor='black')
                    t0=time.time()
                    while True:
                        t=time.time()-t0
                        if t>=duration_s: break
                        r=r0+(r1-r0)*(t/duration_s); dot.radius=r; dot.draw(); self._pp_win.flip()
                    print("[Stim] Done (PsychoPy)"); return
            except Exception as e: print(f"[Stim] PsychoPy error: {e} → OpenCV fallback")
        try:
            self._cv_window(screen_idx,bg_grey)
            size=self._cv_size; bg=int(max(0,min(255,int(bg_grey*255)))); t0=time.time()
            while True:
                t=time.time()-t0
                if t>=duration_s: break
                r=int(r0+(r1-r0)*(t/duration_s)); frame=np.full((size[1],size[0],3),bg,dtype=np.uint8)
                if HAVE_OPENCV:
                    cv2.circle(frame,(size[0]//2,size[1]//2),r,(0,0,0),-1); cv2.imshow(self._cv_window_name,frame)
                    if cv2.waitKey(1) & 0xFF==27: break
            print("[Stim] Done (OpenCV)")
        except Exception as e: print(f"[Stim] Fallback display unavailable: {e}"); wait_s(duration_s); print("[Stim] Done (timing only)")
    def close(self):
        try:
            if self._pp_win is not None: self._pp_win.close()
        except Exception: pass
        self._pp_win=None; self._pp_cfg=None
        if self._cv_open and HAVE_OPENCV:
            try: cv2.destroyWindow(self._cv_window_name)
            except Exception: pass
            self._cv_open=False

# -------------------- Trial Runner --------------------
class TrialRunner:
    def __init__(self,cfg:Config,hw:HardwareBridge,cam0:'CameraNode',cam1:'CameraNode',log_path:str):
        self.cfg=cfg; self.hw=hw; self.cam0=cam0; self.cam1=cam1; self.stim=LoomingStim(cfg)
        ensure_dir(os.path.dirname(log_path) or "."); new=not os.path.exists(log_path)
        self.log=open(log_path,"a",newline="",encoding="utf-8"); self.csvw=csv.writer(self.log)
        if new:
            self.csvw.writerow(["timestamp","trial_idx","cam0_path","cam1_path","record_duration_s",
                                "lights_delay_s","stim_delay_s","stim_duration_s","stim_screen_index","stim_fullscreen",
                                "cam0_backend","cam0_ident","cam0_target_fps","cam0_w","cam0_h","cam0_exp_us","cam0_hwtrig",
                                "cam1_backend","cam1_ident","cam1_target_fps","cam1_w","cam1_h","cam1_exp_us","cam1_hwtrig",
                                "video_preset_id","fourcc"])
        self.trial_idx=0
    def close(self):
        try: self.log.close()
        except Exception: pass
        try: self.stim.close()
        except Exception: pass
    def _ext_for_fourcc(self,fourcc:str)->str:
        if fourcc.lower() in ("mp4v","avc1","h264"): return "mp4"
        if fourcc.lower() in ("mjpg","xvid","divx"): return "avi"
        return "mp4"
    def _trial_folder(self)->str:
        base=day_folder(self.cfg.output_root); p=os.path.join(base,f"trial_{now_stamp()}"); ensure_dir(p); return p
    def _record_both(self,folder:str):
        fourcc=self.cfg.fourcc; ext=self._ext_for_fourcc(fourcc)
        out0=os.path.join(folder,f"cam0.{ext}"); out1=os.path.join(folder,f"cam1.{ext}")
        res={"c0": None, "c1": None}
        def rec(cam:CameraNode,pth:str,key:str):
            res[key]=cam.record_clip(pth, float(self.cfg.record_duration_s), fourcc, async_writer=self.cfg.cam_async_writer)
        t0=threading.Thread(target=rec, args=(self.cam0,out0,"c0"))
        t1=threading.Thread(target=rec, args=(self.cam1,out1,"c1"))
        t0.start(); t1.start(); t0.join(); t1.join()
        return res["c0"], res["c1"]
    def run_one(self):
        folder=self._trial_folder()
        self.hw.mark_start()
        print("[Trial] Recording…")
        c0,c1=self._record_both(folder)
        if self.cfg.lights_delay_s>0: print(f"[Trial] Wait {self.cfg.lights_delay_s:.3f}s → LIGHTS ON"); wait_s(self.cfg.lights_delay_s)
        self.hw.lights_on()
        if self.cfg.stim_delay_s>0: print(f"[Trial] Wait {self.cfg.stim_delay_s:.3f}s → STIM"); wait_s(self.cfg.stim_delay_s)
        self.stim.run(self.cfg.stim_duration_s,self.cfg.stim_r0_px,self.cfg.stim_r1_px,self.cfg.stim_bg_grey,self.cfg.stim_screen_index,self.cfg.stim_fullscreen)
        self.hw.lights_off(); self.hw.mark_end()
        self.trial_idx+=1
        self.csvw.writerow([
            now_stamp(), self.trial_idx,
            c0 or "", c1 or "",
            float(self.cfg.record_duration_s),
            float(self.cfg.lights_delay_s), float(self.cfg.stim_delay_s), float(self.cfg.stim_duration_s),
            int(self.cfg.stim_screen_index), bool(self.cfg.stim_fullscreen),
            self.cam0.backend, self.cam0.ident, int(self.cam0.target_fps), self.cam0.adv.get("width",0), self.cam0.adv.get("height",0), self.cam0.adv.get("exposure_us",0), self.cam0.adv.get("hw_trigger",False),
            self.cam1.backend, self.cam1.ident, int(self.cam1.target_fps), self.cam1.adv.get("width",0), self.cam1.adv.get("height",0), self.cam1.adv.get("exposure_us",0), self.cam1.adv.get("hw_trigger",False),
            self.cfg.video_preset_id, self.cfg.fourcc
        ])
        self.log.flush()
        print("[Trial] Logged")

# Patch onto CameraNode: clip recording (kept minimal; MJPG default)
def _record_clip(self:'CameraNode', path:str, seconds:float, fourcc:str="MJPG", async_writer:bool=True)->str:
    ensure_dir(os.path.dirname(path) or ".")
    # Simple writer
    if self.synthetic or self.dev is None or not HAVE_OPENCV:
        t0=time.time()
        while time.time()-t0<seconds: time.sleep(0.01)
        return ""
    w,h=self.dev.frame_size()
    writer=cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), max(1.0,self.target_fps), (w,h))
    t0=time.time(); frames=0
    try:
        if hasattr(self.dev,'start_acquisition'): self.dev.start_acquisition()
        while time.time()-t0<seconds:
            img=self.dev.get_frame()
            if img is not None:
                if img.shape[1]!=w or img.shape[0]!=h:
                    img=cv2.resize(img,(w,h))
                writer.write(img); frames+=1
            else:
                time.sleep(0.001)
    finally:
        try: writer.release()
        except Exception: pass
        if hasattr(self.dev,'stop_acquisition'): self.dev.stop_acquisition()
    print(f"[{self.name}] Wrote {frames} frames → {path}")
    return path
CameraNode.record_clip=_record_clip

# -------------------- GUI --------------------
class SettingsGUI(QtWidgets.QWidget):
    start_experiment=QtCore.pyqtSignal(); stop_experiment=QtCore.pyqtSignal(); apply_settings=QtCore.pyqtSignal(); manual_trigger=QtCore.pyqtSignal()
    probe_requested=QtCore.pyqtSignal()
    refresh_devices=QtCore.pyqtSignal()

    def __init__(self,cfg:Config,cam0:CameraNode,cam1:CameraNode):
        super().__init__()
        self.cfg=cfg; self.cam0=cam0; self.cam1=cam1
        self.setWindowTitle(f"FlyPy — v{__version__}")

        outer=QtWidgets.QVBoxLayout(self)
        scroll=QtWidgets.QScrollArea(self); scroll.setWidgetResizable(True)
        pane=QtWidgets.QWidget(); root=QtWidgets.QVBoxLayout(pane)

        # --- Top row: Preset / Probe / Device Refresh ---
        row0=QtWidgets.QHBoxLayout()
        row0.addWidget(QtWidgets.QLabel("Quick Preset:"))
        self.cb_preset=QtWidgets.QComboBox()
        self.cb_preset.addItem("Blackfly 522 fps (Mono8, 640×512)")
        self.cb_preset.addItem("Blackfly 300 fps (Mono8, 720×540)")
        self.cb_preset.addItem("OpenCV baseline (laptop cam)")
        row0.addWidget(self.cb_preset)
        self.bt_apply_preset=QtWidgets.QPushButton("Apply Preset")
        row0.addWidget(self.bt_apply_preset)
        self.bt_probe=QtWidgets.QPushButton("Probe Max FPS")
        row0.addWidget(self.bt_probe)
        self.bt_refresh=QtWidgets.QPushButton("Refresh Cameras")
        row0.addWidget(self.bt_refresh)
        root.addLayout(row0)
        self.bt_apply_preset.clicked.connect(self._apply_selected_preset)

        # --- Controls row ---
        row=QtWidgets.QHBoxLayout()
        self.bt_start=QtWidgets.QPushButton("Start")
        self.bt_stop=QtWidgets.QPushButton("Stop")
        self.bt_trig=QtWidgets.QPushButton("Trigger Once")
        self.bt_apply=QtWidgets.QPushButton("Apply Settings")
        row.addWidget(self.bt_start); row.addWidget(self.bt_stop); row.addWidget(self.bt_trig); row.addWidget(self.bt_apply)
        root.addLayout(row)
        self.bt_start.clicked.connect(self.start_experiment.emit)
        self.bt_stop.clicked.connect(self.stop_experiment.emit)
        self.bt_trig.clicked.connect(self.manual_trigger.emit)
        self.bt_apply.clicked.connect(self.apply_settings.emit)
        self.bt_probe.clicked.connect(self._probe_clicked)
        self.bt_refresh.clicked.connect(self.refresh_devices.emit)

        self.lbl_status=QtWidgets.QLabel("Status: Idle."); root.addWidget(self.lbl_status)

        grid=QtWidgets.QGridLayout(); root.addLayout(grid)

        # General group
        gen=QtWidgets.QGroupBox("General")
        gl=QtWidgets.QFormLayout(gen)
        self.cb_sim=QtWidgets.QCheckBox("Simulation Mode (timer triggers)"); self.cb_sim.setChecked(self.cfg.simulation_mode); gl.addRow(self.cb_sim)
        self.sb_sim=QtWidgets.QDoubleSpinBox(); self.sb_sim.setRange(0.1,3600.0); self.sb_sim.setDecimals(2); self.sb_sim.setValue(self.cfg.sim_trigger_interval); gl.addRow("Interval between simulated triggers (s):", self.sb_sim)
        self.le_root=QtWidgets.QLineEdit(self.cfg.output_root); btn_browse=QtWidgets.QPushButton("Browse…")
        rowr=QtWidgets.QHBoxLayout(); rowr.addWidget(self.le_root); rowr.addWidget(btn_browse); gl.addRow("Output folder:", rowr)
        self.cb_fmt=QtWidgets.QComboBox(); self._id_by_idx={}; current=0
        for i,p in enumerate(VIDEO_PRESETS):
            self.cb_fmt.addItem(p["label"]); self.cb_fmt.setItemData(i,p["id"]); self._id_by_idx[i]=p["id"]
            if p["id"]==self.cfg.video_preset_id: current=i
        self.cb_fmt.setCurrentIndex(current); gl.addRow("Video format / codec:", self.cb_fmt)
        self.sb_rec=QtWidgets.QDoubleSpinBox(); self.sb_rec.setRange(0.1,600.0); self.sb_rec.setDecimals(2); self.sb_rec.setValue(self.cfg.record_duration_s); gl.addRow("Recording duration (s):", self.sb_rec)
        grid.addWidget(gen, 0, 0, 1, 2)

        # Stimulus group
        stim=QtWidgets.QGroupBox("Stimulus & Timing")
        sl=QtWidgets.QFormLayout(stim)
        self.sb_stim_dur=QtWidgets.QDoubleSpinBox(); self.sb_stim_dur.setRange(0.1,30.0); self.sb_stim_dur.setDecimals(2); self.sb_stim_dur.setValue(self.cfg.stim_duration_s)
        self.sb_r0=QtWidgets.QSpinBox(); self.sb_r0.setRange(1,2000); self.sb_r0.setValue(self.cfg.stim_r0_px)
        self.sb_r1=QtWidgets.QSpinBox(); self.sb_r1.setRange(1,4000); self.sb_r1.setValue(self.cfg.stim_r1_px)
        self.sb_bg=QtWidgets.QDoubleSpinBox(); self.sb_bg.setRange(0.0,1.0); self.sb_bg.setSingleStep(0.05); self.sb_bg.setValue(self.cfg.stim_bg_grey)
        self.sb_light_delay=QtWidgets.QDoubleSpinBox(); self.sb_light_delay.setRange(0.0,10.0); self.sb_light_delay.setDecimals(3); self.sb_light_delay.setValue(self.cfg.lights_delay_s)
        self.sb_stim_delay=QtWidgets.QDoubleSpinBox(); self.sb_stim_delay.setRange(0.0,10.0); self.sb_stim_delay.setDecimals(3); self.sb_stim_delay.setValue(self.cfg.stim_delay_s)
        sl.addRow("Stimulus duration (s):", self.sb_stim_dur)
        sl.addRow("Start radius (px):", self.sb_r0)
        sl.addRow("Final radius (px):", self.sb_r1)
        sl.addRow("Background shade (0–1):", self.sb_bg)
        sl.addRow("Delay: record → lights ON (s):", self.sb_light_delay)
        sl.addRow("Delay: record → stimulus ON (s):", self.sb_stim_delay)
        grid.addWidget(stim, 1, 0, 1, 2)

        # Display & Windows group
        disp=QtWidgets.QGroupBox("Display & Windows")
        dl=QtWidgets.QFormLayout(disp)
        self.cb_stim_screen=QtWidgets.QComboBox(); self.cb_gui_screen=QtWidgets.QComboBox()
        for i,s in enumerate(QtGui.QGuiApplication.screens()):
            g=s.geometry(); label=f"Screen {i} — {g.width()}×{g.height()} @({g.x()},{g.y()})"
            self.cb_stim_screen.addItem(label); self.cb_gui_screen.addItem(label)
        self.cb_stim_screen.setCurrentIndex(min(self.cfg.stim_screen_index, self.cb_stim_screen.count()-1))
        self.cb_gui_screen.setCurrentIndex(min(self.cfg.gui_screen_index, self.cb_gui_screen.count()-1))
        self.cb_full=QtWidgets.QCheckBox("Stimulus fullscreen"); self.cb_full.setChecked(self.cfg.stim_fullscreen)
        self.cb_prewarm=QtWidgets.QCheckBox("Pre-warm stimulus window at launch"); self.cb_prewarm.setChecked(self.cfg.prewarm_stim)
        dl.addRow("Stimulus display screen:", self.cb_stim_screen)
        dl.addRow("GUI display screen:", self.cb_gui_screen)
        dl.addRow(self.cb_full)
        dl.addRow(self.cb_prewarm)
        grid.addWidget(disp, 2, 0, 1, 2)

        # Camera panels with discovery + advanced fields
        self.cam_boxes=[]
        for idx, node in enumerate((cam0, cam1)):
            gb=QtWidgets.QGroupBox(f"Camera {idx}")
            glb=QtWidgets.QGridLayout(gb)

            preview=QtWidgets.QLabel(); preview.setFixedSize(360,240); preview.setStyleSheet("background:#ddd;border:1px solid #aaa;"); preview.setAlignment(QtCore.Qt.AlignCenter)
            glb.addWidget(preview,0,0,8,1)

            cb_backend=QtWidgets.QComboBox(); cb_backend.addItem("OpenCV"); cb_backend.addItem("PySpin")
            cb_backend.setCurrentIndex(0 if node.backend=="OpenCV" else 1)
            glb.addWidget(QtWidgets.QLabel("Backend:"),0,1); glb.addWidget(cb_backend,0,2)

            le_ident=QtWidgets.QLineEdit(node.ident); le_ident.setPlaceholderText("OpenCV index (e.g., 0) OR PySpin serial (e.g., 24102007) OR idx=1")
            glb.addWidget(QtWidgets.QLabel("Device ID:"),1,1); glb.addWidget(le_ident,1,2)

            # Discovery dropdowns
            cb_spin=QtWidgets.QComboBox(); cb_cv=QtWidgets.QComboBox()
            cb_spin.addItem("—"); cb_cv.addItem("—")
            glb.addWidget(QtWidgets.QLabel("PySpin devices:"),2,1); glb.addWidget(cb_spin,2,2)
            glb.addWidget(QtWidgets.QLabel("OpenCV devices:"),3,1); glb.addWidget(cb_cv,3,2)

            # bind to fill ident/backend when pick
            def make_spin_slot(le=le_ident, be=cb_backend, combo=cb_spin):
                def _slot(_i):
                    data=combo.itemData(combo.currentIndex())
                    if data:
                        be.setCurrentIndex(1)  # PySpin
                        le.setText(data)       # serial preferred or idx=N
                return _slot
            cb_spin.currentIndexChanged.connect(make_spin_slot())

            def make_cv_slot(le=le_ident, be=cb_backend, combo=cb_cv):
                def _slot(_i):
                    data=combo.itemData(combo.currentIndex())
                    if data is not None:
                        be.setCurrentIndex(0)  # OpenCV
                        le.setText(str(data))  # numeric index
                return _slot
            cb_cv.currentIndexChanged.connect(make_cv_slot())

            sb_fps=QtWidgets.QSpinBox(); sb_fps.setRange(1,10000); sb_fps.setValue(int(node.target_fps))
            glb.addWidget(QtWidgets.QLabel("Target FPS:"),4,1); glb.addWidget(sb_fps,4,2)

            sb_w=QtWidgets.QSpinBox(); sb_w.setRange(0,10000); sb_w.setSingleStep(2); sb_w.setValue(int(node.adv.get("width",640) or 640))
            sb_h=QtWidgets.QSpinBox(); sb_h.setRange(0,10000); sb_h.setSingleStep(2); sb_h.setValue(int(node.adv.get("height",512) or 512))
            glb.addWidget(QtWidgets.QLabel("ROI Width (0=max):"),5,1); glb.addWidget(sb_w,5,2)
            glb.addWidget(QtWidgets.QLabel("ROI Height (0=max):"),6,1); glb.addWidget(sb_h,6,2)

            sb_exp=QtWidgets.QSpinBox(); sb_exp.setRange(20, 1000000); sb_exp.setSingleStep(50); sb_exp.setValue(int(node.adv.get("exposure_us",1500) or 1500))
            glb.addWidget(QtWidgets.QLabel("Exposure (µs):"),7,1); glb.addWidget(sb_exp,7,2)

            cb_hwtrig=QtWidgets.QCheckBox("Hardware trigger (Line0)"); cb_hwtrig.setChecked(bool(node.adv.get("hw_trigger", False)))
            glb.addWidget(cb_hwtrig,8,1,1,2)

            # Live preview
            bt_live=QtWidgets.QPushButton("Live Preview (high-FPS)")
            glb.addWidget(bt_live,9,1,1,2)

            lbl_rep=QtWidgets.QLabel("Driver-reported FPS: ~0.0"); glb.addWidget(lbl_rep,10,1,1,2)

            # connect live
            def make_live_slot(node=node):
                def _slot(): node.start_live_preview()
                return _slot
            bt_live.clicked.connect(make_live_slot())

            self.cam_boxes.append({
                "gb":gb,"preview":preview,"cb_backend":cb_backend,"le_ident":le_ident,"cb_spin":cb_spin,"cb_cv":cb_cv,
                "sb_fps":sb_fps,"sb_w":sb_w,"sb_h":sb_h,"sb_exp":sb_exp,"cb_hw":cb_hwtrig,"lbl_rep":lbl_rep})
            grid.addWidget(gb, 3+idx, 0, 1, 2)

        btn_browse.clicked.connect(self._browse)

        scroll.setWidget(pane); outer.addWidget(scroll)
        self._update_footer=QtWidgets.QLabel("Preset tip: Blackfly 522 fps uses PySpin + Mono8 + ROI 640×512, Exposure ~1500 µs, MJPG to NVMe.")
        outer.addWidget(self._update_footer)

    def _apply_selected_preset(self):
        idx=self.cb_preset.currentIndex()
        for i in range(self.cb_fmt.count()):
            if (self.cb_fmt.itemData(i) or "").lower()=="avi_mjpg":
                self.cb_fmt.setCurrentIndex(i); break
        if idx==0:
            for box in self.cam_boxes:
                box["cb_backend"].setCurrentIndex(1)
                box["sb_fps"].setValue(522)
                box["sb_w"].setValue(640); box["sb_h"].setValue(512)
                box["sb_exp"].setValue(1500); box["cb_hw"].setChecked(False)
        elif idx==1:
            for box in self.cam_boxes:
                box["cb_backend"].setCurrentIndex(1)
                box["sb_fps"].setValue(300)
                box["sb_w"].setValue(720); box["sb_h"].setValue(540)
                box["sb_exp"].setValue(2500); box["cb_hw"].setChecked(False)
        else:
            for n,box in enumerate(self.cam_boxes):
                box["cb_backend"].setCurrentIndex(0)
                box["le_ident"].setText(str(n))
                box["sb_fps"].setValue(60)
                box["sb_w"].setValue(640); box["sb_h"].setValue(480)
                box["sb_exp"].setValue(5000); box["cb_hw"].setChecked(False)
        self.sb_rec.setValue(2.0)
        self.apply_settings.emit()

    def _probe_clicked(self):
        self.probe_requested.emit()

    def _browse(self):
        d=QtWidgets.QFileDialog.getExistingDirectory(self,"Select output folder", self.le_root.text() or os.getcwd())
        if d: self.le_root.setText(d)

    def set_preview_image(self, cam_idx:int, img_rgb: np.ndarray):
        if img_rgb is None: return
        h,w,_=img_rgb.shape
        qimg=QtGui.QImage(img_rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        pix=QtGui.QPixmap.fromImage(qimg)
        self.cam_boxes[cam_idx]["preview"].setPixmap(pix)

    def update_cam_fps_labels(self, f0:float, f1:float):
        self.cam_boxes[0]["lbl_rep"].setText(f"Driver-reported FPS: ~{f0:.1f}")
        self.cam_boxes[1]["lbl_rep"].setText(f"Driver-reported FPS: ~{f1:.1f}")

    # device lists update
    def populate_device_lists(self, spin_list:List[Dict[str,Any]], cv_list:List[Dict[str,Any]]):
        for box in self.cam_boxes:
            cb_spin=box["cb_spin"]; cb_cv=box["cb_cv"]
            cb_spin.blockSignals(True); cb_cv.blockSignals(True)
            cb_spin.clear(); cb_cv.clear()
            if spin_list:
                cb_spin.addItem("—", None)
                for d in spin_list:
                    label=f"idx={d['index']} — {d.get('model','?')} — SN {d.get('serial','?') or '?'}"
                    ident = d.get('serial') or f"idx={d['index']}"
                    cb_spin.addItem(label, ident)
            else:
                cb_spin.addItem("(no PySpin devices)", None)
            if cv_list:
                cb_cv.addItem("—", None)
                for d in cv_list:
                    label=f"idx={d['index']} — {d.get('size','')}"
                    cb_cv.addItem(label, d['index'])
            else:
                cb_cv.addItem("(no OpenCV devices)", None)
            cb_spin.blockSignals(False); cb_cv.blockSignals(False)

# -------------------- Main App --------------------
class MainApp(QtWidgets.QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.cfg=Config()
        try:
            import argparse
            ap=argparse.ArgumentParser(add_help=False)
            ap.add_argument("--simulate",action="store_true")
            ap.add_argument("--prewarm-stim",dest="prewarm_stim",action="store_true")
            ns,_=ap.parse_known_args(argv[1:])
            if ns.simulate: self.cfg.simulation_mode=True
            if getattr(ns,"prewarm_stim",False): self.cfg.prewarm_stim=True
        except Exception: pass

        self.hw=HardwareBridge(self.cfg)
        self.cam0=CameraNode("cam0", self.cfg.cam0_backend, self.cfg.cam0_id, self.cfg.cam0_target_fps,
                             adv={"width":self.cfg.cam0_width,"height":self.cfg.cam0_height,"exposure_us":self.cfg.cam0_exposure_us,"hw_trigger":self.cfg.cam0_hw_trigger})
        self.cam1=CameraNode("cam1", self.cfg.cam1_backend, self.cfg.cam1_id, self.cfg.cam1_target_fps,
                             adv={"width":self.cfg.cam1_width,"height":self.cfg.cam1_height,"exposure_us":self.cfg.cam1_exposure_us,"hw_trigger":self.cfg.cam1_hw_trigger})

        ensure_dir(self.cfg.output_root)
        log_path=os.path.join(self.cfg.output_root,"trials_log.csv")
        self.runner=TrialRunner(self.cfg,self.hw,self.cam0,self.cam1,log_path)

        self.gui=SettingsGUI(self.cfg,self.cam0,self.cam1)

        # Connect GUI signals
        self.gui.start_experiment.connect(self.start_loop)
        self.gui.stop_experiment.connect(self.stop_loop)
        self.gui.apply_settings.connect(self.apply_from_gui)
        self.gui.manual_trigger.connect(self.trigger_once)
        self.gui.probe_requested.connect(self.start_probe)
        self.gui.refresh_devices.connect(self.refresh_devices)

        self.show_scaled_gui(self.cfg.gui_screen_index)

        self.running=False; self.in_trial=False; self.thread=None
        self.preview_timer=QtCore.QTimer(self); self.preview_timer.setInterval(500); self.preview_timer.timeout.connect(self.update_previews); self.preview_timer.start()

        self.aboutToQuit.connect(self.cleanup); atexit.register(self.cleanup)

        # Initial devices scan
        QtCore.QTimer.singleShot(200, self.refresh_devices)

        if self.cfg.prewarm_stim:
            QtCore.QTimer.singleShot(300, lambda: self.runner.stim.open_persistent(self.cfg.stim_screen_index,self.cfg.stim_fullscreen,self.cfg.stim_bg_grey))

        self._print_startup_summary()

    def _print_startup_summary(self):
        print("=== FlyPy Startup ===")
        print(f"Version: {__version__}")
        print(f"OpenCV: {'OK' if HAVE_OPENCV else 'MISSING'}")
        print(f"PsychoPy: {'will use' if _ensure_psychopy_loaded() else 'not available (OpenCV fallback)'}")
        try:
            import PySpin as _ps; print("PySpin: OK")
        except Exception as e:
            print(f"PySpin: MISSING ({e}) — install Spinnaker SDK + PySpin and ensure PATH contains bin64/lib64.")
        print("======================")

    def show_scaled_gui(self, screen_index:int):
        screens=QtGui.QGuiApplication.screens()
        geo=screens[screen_index].availableGeometry() if 0<=screen_index<len(screens) else QtGui.QGuiApplication.primaryScreen().availableGeometry()
        target_w=max(980,int(geo.width()*0.9)); target_h=max(700,int(geo.height()*0.9))
        target_w=min(target_w,geo.width()); target_h=min(target_h,geo.height())
        self.gui.resize(target_w,target_h)
        x=geo.x()+(geo.width()-target_w)//2; y=geo.y()+(geo.height()-target_h)//2; self.gui.move(x,y); self.gui.show()

    def position_gui(self, screen_index:int):
        screens=QtGui.QGuiApplication.screens()
        if 0<=screen_index<len(screens):
            geo=screens[screen_index].availableGeometry()
            w=min(self.gui.width(), geo.width()); h=min(self.gui.height(), geo.height())
            x=geo.x()+(geo.width()-w)//2; y=geo.y()+(geo.height()-h)//2
            self.gui.resize(w,h); self.gui.move(x,y); self.gui.show()
        else:
            self.gui.show()

    def apply_from_gui(self):
        prev_sim=self.cfg.simulation_mode
        self.cfg.simulation_mode=bool(self.gui.cb_sim.isChecked())
        self.cfg.sim_trigger_interval=float(self.gui.sb_sim.value())
        self.cfg.output_root=self.gui.le_root.text().strip() or self.cfg.output_root

        idx=self.gui.cb_fmt.currentIndex()
        preset_id=self.gui.cb_fmt.itemData(idx) or "avi_mjpg"
        self.cfg.video_preset_id=preset_id; self.cfg.fourcc=PRESETS_BY_ID[preset_id]["fourcc"]
        self.cfg.record_duration_s=float(self.gui.sb_rec.value())

        self.cfg.stim_duration_s=float(self.gui.sb_stim_dur.value())
        self.cfg.stim_r0_px=int(self.gui.sb_r0.value())
        self.cfg.stim_r1_px=int(self.gui.sb_r1.value())
        self.cfg.stim_bg_grey=float(self.gui.sb_bg.value())
        self.cfg.lights_delay_s=float(self.gui.sb_light_delay.value())
        self.cfg.stim_delay_s=float(self.gui.sb_stim_delay.value())

        self.cfg.stim_screen_index=int(self.gui.cb_stim_screen.currentIndex())
        self.cfg.gui_screen_index=int(self.gui.cb_gui_screen.currentIndex())
        self.cfg.stim_fullscreen=bool(self.gui.cb_full.isChecked())
        self.cfg.prewarm_stim=bool(self.gui.cb_prewarm.isChecked())

        for i,node in enumerate((self.cam0,self.cam1)):
            box=self.gui.cam_boxes[i]
            backend="OpenCV" if box["cb_backend"].currentIndex()==0 else "PySpin"
            ident=box["le_ident"].text().strip() or ("0" if i==0 else "1")
            fps=int(box["sb_fps"].value())
            adv={
                "width": int(box["sb_w"].value() or 0),
                "height": int(box["sb_h"].value() or 0),
                "exposure_us": int(box["sb_exp"].value() or 0),
                "hw_trigger": bool(box["cb_hw"].isChecked())
            }
            node.set_backend_ident(backend, ident, adv=adv)
            node.set_target_fps(fps)
            if i==0:
                self.cfg.cam0_backend, self.cfg.cam0_id, self.cfg.cam0_target_fps = backend, ident, fps
                self.cfg.cam0_width, self.cfg.cam0_height = adv["width"], adv["height"]
                self.cfg.cam0_exposure_us, self.cfg.cam0_hw_trigger = adv["exposure_us"], adv["hw_trigger"]
            else:
                self.cfg.cam1_backend, self.cfg.cam1_id, self.cfg.cam1_target_fps = backend, ident, fps
                self.cfg.cam1_width, self.cfg.cam1_height = adv["width"], adv["height"]
                self.cfg.cam1_exposure_us, self.cfg.cam1_hw_trigger = adv["exposure_us"], adv["hw_trigger"]

        if prev_sim!=self.cfg.simulation_mode:
            if self.cfg.simulation_mode: self.hw.close(); self.hw.simulated=True
            else: self.hw.simulated=False; self.hw._opened=False; self.hw.ser=None

        ensure_dir(self.cfg.output_root)
        self.gui.lbl_status.setText("Status: Settings applied.")
        print("[Main] Settings applied")

        if self.cfg.prewarm_stim:
            self.runner.stim.open_persistent(self.cfg.stim_screen_index,self.cfg.stim_fullscreen,self.cfg.stim_bg_grey)
        else:
            try: self.runner.stim.close()
            except Exception: pass

    def update_previews(self):
        if self.in_trial:
            self.gui.lbl_status.setText("Status: Trial running (preview paused)")
            self.gui.update_cam_fps_labels(self.cam0.driver_fps(), self.cam1.driver_fps())
            return
        p0=self.gui.cam_boxes[0]["preview"]; p1=self.gui.cam_boxes[1]["preview"]
        w0,h0=p0.width(),p0.height(); w1,h1=p1.width(),p1.height()
        img0=self.cam0.grab_preview(w0,h0); img1=self.cam1.grab_preview(w1,h1)
        self.gui.set_preview_image(0,img0); self.gui.set_preview_image(1,img1)
        self.gui.update_cam_fps_labels(self.cam0.driver_fps(), self.cam1.driver_fps())
        self.gui.lbl_status.setText("Status: Waiting / Idle.")

    def loop(self):
        self.gui.lbl_status.setText("Status: Watching for triggers…"); print("[Main] Trigger loop started")
        try:
            while self.running:
                if self.hw.check_trigger():
                    self.in_trial=True
                    self.gui.lbl_status.setText("Status: Triggered — running trial…")
                    try:
                        self.runner.run_one()
                    except Exception as e:
                        print(f"[Main] Trial error: {e}")
                    self.in_trial=False
                    self.gui.lbl_status.setText("Status: Waiting / Idle.")
                QtWidgets.QApplication.processEvents()
                time.sleep(0.005)
        finally:
            print("[Main] Trigger loop exiting")

    def start_loop(self):
        if self.running: return
        self.running=True
        self.thread=threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        self.gui.lbl_status.setText("Status: Trigger loop running.")
        print("[Main] Start")

    def stop_loop(self):
        if not self.running: return
        self.running=False
        if self.thread:
            self.thread.join(); self.thread=None
        self.gui.lbl_status.setText("Status: Stopped.")
        print("[Main] Stop")

    def trigger_once(self):
        if self.in_trial: return
        self.in_trial=True
        try:
            self.runner.run_one()
        except Exception as e:
            print(f"[Main] Manual trial error: {e}")
        self.in_trial=False

    # ---- Probe Max FPS ----
    def start_probe(self):
        try: self.apply_from_gui()
        except Exception as e: print(f"[Probe] apply_from_gui failed: {e}")
        self.preview_timer.stop()
        self.gui.lbl_status.setText("Status: Probing max FPS…")
        def worker():
            try:
                res0=self.cam0.probe_max_fps(3.0)
                res1=self.cam1.probe_max_fps(3.0)
                txt=(
                    f"Probe window: 3.0 s\n\n"
                    f"Camera 0 → FPS: {res0[0]:.1f}  (frames={res0[1]}, drops={res0[2]})\n"
                    f"Camera 1 → FPS: {res1[0]:.1f}  (frames={res1[1]}, drops={res1[2]})\n\n"
                    f"Tip: set Target FPS to ~90% of the measured value for stability."
                )
            except Exception as e:
                txt=f"Probe failed: {e}"
            QtCore.QMetaObject.invokeMethod(self, "_finish_probe", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, txt))
        threading.Thread(target=worker, daemon=True).start()

    @QtCore.pyqtSlot(str)
    def _finish_probe(self, msg:str):
        self.preview_timer.start()
        self.gui.lbl_status.setText("Status: Probe finished.")
        QtWidgets.QMessageBox.information(self.gui, "Probe Max FPS", msg)

    # ---- Device scanning ----
    def refresh_devices(self):
        # PySpin list
        spin_devices=_spin_list_devices()
        # OpenCV list (best-effort indices 0..9)
        cv_devices=[]
        if HAVE_OPENCV:
            for i in range(10):
                cap=None
                for be in ([cv2.CAP_DSHOW,cv2.CAP_MSMF,cv2.CAP_ANY] if os.name=="nt" else [cv2.CAP_ANY]):
                    try:
                        cap=cv2.VideoCapture(i, be)
                        if cap and cap.isOpened():
                            w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                            cv_devices.append({"index":i,"size":f"{w}x{h}"})
                            break
                    except Exception:
                        pass
                    finally:
                        if cap: cap.release()
        self.gui.populate_device_lists(spin_devices, cv_devices)
        print(f"[Scan] PySpin={len(spin_devices)} device(s); OpenCV={len(cv_devices)} device(s)")

    def cleanup(self):
        print("[Main] Cleanup…")
        try: self.hw.close()
        except Exception: pass
        for node in (self.cam0, self.cam1):
            try: node.release()
            except Exception: pass
        try: self.runner.stim.close()
        except Exception: pass
        print("[Main] Cleanup done")

def main():
    print("[BOOT] PyQt5 imported OK")
    try:
        from PyQt5 import QtCore as _qtc  # noqa: F401
        plugin_path = QtWidgets.QLibraryInfo.location(QtWidgets.QLibraryInfo.PluginsPath)
        print(f"[BOOT] Qt plugin path = {plugin_path}")
        try:
            fmt = QtGui.QOpenGLContext.globalShareContext().format()
            print(f"[BOOT] Qt OpenGL backend = angle")
        except Exception:
            pass
    except Exception:
        pass
    app=MainApp(sys.argv); sys.exit(app.exec_())

if __name__=="__main__": 
    main()
