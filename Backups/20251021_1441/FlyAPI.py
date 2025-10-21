# FlyAPI.py
# FlyPy — Unified Trigger → Cameras + Lights + Looming Stimulus
# v1.35.4
# - FIX: PySpin device selection now matches DeviceSerialNumber (not UniqueID),
#        so each serial (e.g., 24102007 vs 24102017) opens the correct camera.
# - FIX: Handle "BeginAcquisition: Camera is already streaming" once, avoid spam.
# - FIX: Advanced… toggle handler signature accepts clicked(bool), no crash.
# - Keeps: process-lifetime PySpin System; distinct default assignment after Refresh.
# - Keeps: logs under Desktop\LevitatingInsect-main\logs with end timestamp
# - Keeps: preview toggles off by default, full-res recording only, FPS probe
# - Keeps: stimulus start/end size, fullscreen/screen chooser, simulation mode

import os, sys, time, csv, atexit, threading, queue, logging, shutil
from collections import deque
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import importlib
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

__version__ = "1.35.4"

# -------------------- Logging --------------------
LOG_DIR_DEFAULT = r"C:\Users\Murpheylab\Desktop\LevitatingInsect-main\logs"
def ensure_dir(p:str): os.makedirs(p, exist_ok=True)
def now_stamp()->str: return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
def day_stamp()->str: return datetime.now().strftime("%Y%m%d")

def init_logging()->tuple[logging.Logger, str]:
    log_dir = LOG_DIR_DEFAULT if os.name=="nt" else os.path.join(os.getcwd(),"logs")
    ensure_dir(log_dir)
    start_stamp = now_stamp()
    tmp_name = os.path.join(log_dir, f"FlyPy_run_{start_stamp}.log.tmp")
    logger = logging.getLogger("FlyPy")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(tmp_name, encoding="utf-8")
    fh.setFormatter(fmt); fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt); ch.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(fh); logger.addHandler(ch)
    logger.info(f"=== FlyPy Log Start v{__version__} ===")
    return logger, tmp_name

LOGGER, _LOG_TMP_PATH = init_logging()

def finalize_logging():
    try:
        for h in list(LOGGER.handlers):
            try: h.flush(); h.close()
            except Exception: pass
            LOGGER.removeHandler(h)
    except Exception:
        pass
    try:
        end_stamp = now_stamp()
        final_name = _LOG_TMP_PATH.replace(".tmp", f"__ENDED_{end_stamp}.log")
        if os.path.exists(_LOG_TMP_PATH):
            base, ext = os.path.splitext(final_name)
            i=1
            while os.path.exists(final_name):
                final_name = f"{base}_{i}{ext}"; i+=1
            shutil.move(_LOG_TMP_PATH, final_name)
    except Exception:
        pass

def excepthook(exctype, value, tb):
    import traceback
    msg = "".join(traceback.format_exception(exctype, value, tb))
    try: LOGGER.critical("UNCAUGHT EXCEPTION:\n%s", msg)
    except Exception: pass
    sys.__excepthook__(exctype, value, tb)

sys.excepthook = excepthook
atexit.register(finalize_logging)

try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# -------------------- Optional Libraries --------------------
try:
    import cv2; HAVE_OPENCV=True
except Exception as e:
    LOGGER.warning("OpenCV not available: %s", e); HAVE_OPENCV=False; cv2=None  # type: ignore

PSY_LOADED=None; visual=None; core=None
def _ensure_psychopy_loaded()->bool:
    global PSY_LOADED, visual, core
    if PSY_LOADED is True: return True
    if PSY_LOADED is False: return False
    try:
        importlib.import_module("psychopy")
        visual=importlib.import_module("psychopy.visual")
        core  =importlib.import_module("psychopy.core")
        PSY_LOADED=True; LOGGER.info("PsychoPy available"); return True
    except Exception as e:
        LOGGER.info("PsychoPy not available (%s) → using OpenCV fallback", e)
        visual=None; core=None; PSY_LOADED=False; return False

def wait_s(sec:float):
    if _ensure_psychopy_loaded():
        try: core.wait(sec); return
        except Exception: pass
    time.sleep(sec)

def day_folder(root:str)->str:
    d=day_stamp(); path=os.path.join(root,d); ensure_dir(path); return path

try:
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
except Exception:
    pass

# -------------------- Video Presets --------------------
VIDEO_PRESETS=[
    {"id":"avi_mjpg","label":"AVI / MJPG — light CPU, huge files (fast)","fourcc":"MJPG"},
    {"id":"avi_xvid","label":"AVI / XVID — broad compatibility","fourcc":"XVID"},
    {"id":"mp4_mp4v","label":"MP4 / mp4v — very compatible","fourcc":"mp4v"},
]
PRESETS_BY_ID={p["id"]:p for p in VIDEO_PRESETS}
def default_preset_id()->str: return "avi_mjpg"

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

        # Stimulus (growing black dot on white background)
        self.stim_duration_s=1.5
        self.stim_r0_px=8
        self.stim_r1_px=240
        self.stim_bg_grey=1.0
        self.lights_delay_s=0.0
        self.stim_delay_s=0.0
        self.stim_screen_index=0
        self.stim_fullscreen=False
        self.gui_screen_index=0

        # Cameras
        self.cam0_backend="PySpin"; self.cam1_backend="PySpin"
        self.cam0_id=""; self.cam1_id=""
        self.cam0_target_fps=522; self.cam1_target_fps=522
        self.cam0_width=0; self.cam1_width=0
        self.cam0_height=0; self.cam1_height=0
        self.cam0_exposure_us=1500; self.cam1_exposure_us=1500
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
        except Exception as e:
            LOGGER.debug("Port autodetect failed: %s", e)
        return None
    def _open_if_needed(self):
        if self.simulated or self._opened: return
        self._opened=True
        try:
            import serial
            if not self.port: self.port=self._autodetect_port()
            if self.port:
                try:
                    self.ser=serial.Serial(self.port,self.baud,timeout=0.01)
                    wait_s(1.2)
                    LOGGER.info("[HW] Serial open: %s @ %d", self.port, self.baud)
                except Exception as e:
                    LOGGER.warning("[HW] Open failed: %s → simulation", e); self.simulated=True
            else:
                LOGGER.info("[HW] No CH340/UNO port found → simulation"); self.simulated=True
        except Exception as e:
            LOGGER.info("[HW] pyserial not available (%s) → simulation", e); self.simulated=True
    def check_trigger(self)->bool:
        if self.simulated:
            now=time.time()
            if now-self._last_sim>=self.cfg.sim_trigger_interval:
                self._last_sim=now; LOGGER.info("[HW] (Sim) Trigger"); return True
            return False
        self._open_if_needed()
        try:
            if self.ser and self.ser.in_waiting:
                line=self.ser.readline().decode(errors="ignore").strip()
                if line=="T": return True
        except Exception as e: LOGGER.warning("[HW] Read error: %s", e)
        return False
    def _send(self,text:str):
        self._open_if_needed()
        if self.simulated or not self.ser: LOGGER.info("[HW] (Sim) SEND: %s", text); return
        try: self.ser.write((text.strip()+"\n").encode("utf-8",errors="ignore"))
        except Exception as e: LOGGER.warning("[HW] Write error: %s", e)
    def mark_start(self): self._send("MARK START")
    def mark_end(self):   self._send("MARK END")
    def lights_on(self):  self._send("LIGHT ON")
    def lights_off(self): self._send("LIGHT OFF")
    def close(self):
        if not self.simulated and self.ser:
            try: self.ser.close()
            except Exception: pass
        self.ser=None; self._opened=False

# -------------------- Camera Backends --------------------
class BaseCamera:
    def open(self): raise NotImplementedError
    def get_frame(self): raise NotImplementedError
    def release(self): raise NotImplementedError
    def frame_size(self)->Tuple[int,int]: raise NotImplementedError
    def start_acquisition(self): pass
    def stop_acquisition(self): pass

class OpenCVCamera(BaseCamera):
    def __init__(self,index:int,target_fps:float):
        if not HAVE_OPENCV: raise RuntimeError("OpenCV is not installed")
        self.index=index; self.target_fps=float(target_fps); self.cap=None
    def open(self):
        backends=[cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name=="nt" else [cv2.CAP_ANY]
        for be in backends:
            try:
                cap=cv2.VideoCapture(self.index, be)
                if cap and cap.isOpened():
                    try:
                        cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    except Exception: pass
                    self.cap=cap; LOGGER.info("[OpenCV] index %d opened via backend=%s", self.index, be); return
                if cap: cap.release()
            except Exception as e:
                LOGGER.debug("OpenCV backend %s failed: %s", be, e)
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

# -------- PySpin backend (Spinnaker) --------
HAVE_PYSPIN=False; PySpin=None; _SPIN_SYS=None

def _spin_system_get():
    """Get (and keep) the singleton System for the process lifetime."""
    global PySpin, HAVE_PYSPIN, _SPIN_SYS
    if not HAVE_PYSPIN:
        try:
            import PySpin as _ps
            PySpin=_ps; HAVE_PYSPIN=True
        except Exception as e:
            LOGGER.info("[PySpin] Not available: %s", e)
            return None
    if _SPIN_SYS is None:
        _SPIN_SYS = PySpin.System.GetInstance()
    return _SPIN_SYS

def _spin_system_release_final():
    global _SPIN_SYS
    if _SPIN_SYS is not None:
        try:
            _SPIN_SYS.ReleaseInstance()
        except Exception as e:
            LOGGER.warning("[PySpin] ReleaseInstance warning (final): %s", e)
        _SPIN_SYS=None

def spin_enumerate()->List[Dict[str,str]]:
    """Return list of {'serial','model','display'}; safe wrt SWIG refs."""
    sys_inst=_spin_system_get()
    out=[]
    if sys_inst is None: return out
    lst = sys_inst.GetCameras()
    try:
        n = lst.GetSize()
        for i in range(n):
            cam=None; dmap=None
            try:
                cam = lst.GetByIndex(i)
                dmap = cam.GetTLDeviceNodeMap()
                s_serial = None
                try:
                    s_serial = PySpin.CStringPtr(dmap.GetNode("DeviceSerialNumber")).GetValue()
                except Exception:
                    try: s_serial = PySpin.CStringPtr(dmap.GetNode("DeviceID")).GetValue()
                    except Exception:
                        try: s_serial = cam.GetUniqueID()
                        except Exception: s_serial = f"idx={i}"
                try:
                    s_model = PySpin.CStringPtr(dmap.GetNode("DeviceModelName")).GetValue()
                except Exception:
                    s_model = "UnknownModel"
                out.append({"serial":str(s_serial), "model":str(s_model), "display":f"PySpin {s_serial} — {s_model}"})
            except Exception as e:
                LOGGER.debug("[PySpin] enumerate cam %d failed: %s", i, e)
            finally:
                try: dmap = None
                except Exception: pass
                try: cam = None
                except Exception: pass
    finally:
        try: lst.Clear()
        except Exception: pass
        try: del lst
        except Exception: pass
    return out

def _safe_set_enum(nodemap, name, symbolic):
    try:
        node = PySpin.CEnumerationPtr(nodemap.GetNode(name))
        if not PySpin.IsWritable(node): return False
        entry = node.GetEntryByName(symbolic)
        if not PySpin.IsReadable(entry): return False
        node.SetIntValue(entry.GetValue()); return True
    except Exception as e:
        LOGGER.debug("[PySpin] Enum %s=%s failed: %s", name, symbolic, e)
        return False

def _safe_set_float(nodemap, name, value):
    try:
        node = PySpin.CFloatPtr(nodemap.GetNode(name))
        if not PySpin.IsWritable(node): return False
        lo, hi = node.GetMin(), node.GetMax()
        v = max(lo, min(hi, float(value)))
        node.SetValue(v); return True
    except Exception as e:
        LOGGER.debug("[PySpin] Float %s=%s failed: %s", name, value, e)
        return False

def _safe_set_bool(nodemap, name, value:bool):
    try:
        node = PySpin.CBooleanPtr(nodemap.GetNode(name))
        if not PySpin.IsWritable(node): return False
        node.SetValue(bool(value)); return True
    except Exception as e:
        LOGGER.debug("[PySpin] Bool %s=%s failed: %s", name, value, e)
        return False

def _align_to_inc(val, inc, lo, hi):
    if inc <= 0: return int(max(lo, min(hi, val)))
    v = int(val // inc * inc)
    return max(int(lo), min(int(hi), int(v)))

class SpinnakerCamera(BaseCamera):
    def __init__(self, serial:str, target_fps:float, width:int=0, height:int=0, exposure_us:int=1500, hw_trigger:bool=False):
        self.serial = (serial or "").strip()
        self.target_fps = float(target_fps)
        self.req_w = int(width); self.req_h = int(height)
        self.exposure_us = int(exposure_us)
        self.hw_trigger = bool(hw_trigger)
        self.cam = None; self.node = None; self.snode = None
        self._acq = False; self._last_size = (640,480); self._mono = True

    def open(self):
        sys_inst = _spin_system_get()
        if sys_inst is None:
            raise RuntimeError("PySpin not available; install Spinnaker SDK + PySpin and ensure DLLs are in PATH.")
        lst = sys_inst.GetCameras()
        chosen_idx = 0
        try:
            n = lst.GetSize()
            if n==0:
                raise RuntimeError("No Spinnaker cameras detected")
            if self.serial:
                for i in range(n):
                    try:
                        cam_i = lst.GetByIndex(i)
                        dmap = cam_i.GetTLDeviceNodeMap()
                        sn = None
                        try:
                            sn = PySpin.CStringPtr(dmap.GetNode("DeviceSerialNumber")).GetValue()
                        except Exception:
                            try: sn = PySpin.CStringPtr(dmap.GetNode("DeviceID")).GetValue()
                            except Exception:
                                try: sn = cam_i.GetUniqueID()
                                except Exception: sn = None
                        if str(sn) == str(self.serial):
                            chosen_idx = i; break
                    except Exception:
                        pass
            chosen = lst.GetByIndex(chosen_idx)
            self.cam = chosen; self.cam.Init()
            self.node = self.cam.GetNodeMap(); self.snode = self.cam.GetTLStreamNodeMap()

            _safe_set_enum(self.snode, "StreamBufferHandlingMode", "NewestOnly")
            try:
                mode = PySpin.CEnumerationPtr(self.snode.GetNode("StreamBufferCountMode"))
                if PySpin.IsWritable(mode):
                    mode.SetIntValue(mode.GetEntryByName("Manual").GetValue())
                    cnt = PySpin.CIntegerPtr(self.snode.GetNode("StreamBufferCountManual"))
                    if PySpin.IsWritable(cnt): cnt.SetValue(max(int(cnt.GetMin()), min(int(cnt.GetMax()), 64)))
            except Exception as e:
                LOGGER.debug("[PySpin] Stream buffer config note: %s", e)

            ok = _safe_set_enum(self.node, "PixelFormat", "Mono8")
            self._mono = ok
            if not ok:
                ok = _safe_set_enum(self.node, "PixelFormat", "BayerRG8")
                self._mono = False if ok else True

            try:
                w = PySpin.CIntegerPtr(self.node.GetNode("Width"))
                h = PySpin.CIntegerPtr(self.node.GetNode("Height"))
                ox = PySpin.CIntegerPtr(self.node.GetNode("OffsetX"))
                oy = PySpin.CIntegerPtr(self.node.GetNode("OffsetY"))
                maxw = int(w.GetMax()); maxh = int(h.GetMax())
                incw = int(w.GetInc()) or 2; inch = int(h.GetInc()) or 2
                reqw = maxw if self.req_w<=0 or self.req_w>maxw else _align_to_inc(self.req_w, incw, w.GetMin(), maxw)
                reqh = maxh if self.req_h<=0 or self.req_h>maxh else _align_to_inc(self.req_h, inch, h.GetMin(), maxh)
                cx = max(0, (maxw - reqw)//(2*incw)*incw); cy = max(0, (maxh - reqh)//(2*inch)*inch)
                if PySpin.IsWritable(ox): ox.SetValue(cx)
                if PySpin.IsWritable(oy): oy.SetValue(cy)
                w.SetValue(reqw); h.SetValue(reqh)
                self._last_size = (int(reqw), int(reqh))
            except Exception as e:
                LOGGER.info("[PySpin] ROI note: %s", e)

            _safe_set_enum(self.node, "ExposureAuto", "Off")
            if self.exposure_us>0:
                period_us = 1e6/max(1.0, self.target_fps)
                exp_us = min(self.exposure_us, int(period_us*0.85))
                _safe_set_float(self.node, "ExposureTime", exp_us)
            _safe_set_enum(self.node, "GainAuto", "Off")

            if not self.hw_trigger:
                _safe_set_bool(self.node, "AcquisitionFrameRateEnable", True)
                _safe_set_float(self.node, "AcquisitionFrameRate", self.target_fps)
            else:
                _safe_set_bool(self.node, "AcquisitionFrameRateEnable", False)
                _safe_set_enum(self.node, "TriggerMode", "Off")
                _safe_set_enum(self.node, "TriggerSelector", "FrameStart")
                _safe_set_enum(self.node, "TriggerSource", "Line0")
                _safe_set_enum(self.node, "TriggerActivation", "RisingEdge")
                _safe_set_enum(self.node, "TriggerOverlap", "ReadOut")
                _safe_set_enum(self.node, "TriggerMode", "On")

            _safe_set_enum(self.node, "AcquisitionMode", "Continuous")

            try:
                dl = PySpin.CFloatPtr(self.node.GetNode("DeviceLinkThroughputLimit"))
                if dl and PySpin.IsWritable(dl):
                    desired_Bps = self._last_size[0]*self._last_size[1]*(1 if self._mono else 1.5)*self.target_fps
                    dl.SetValue(min(dl.GetMax(), max(dl.GetMin(), desired_Bps*1.2)))
            except Exception:
                pass

            LOGGER.info("[PySpin] open: serial=%s size=%s fps=%.1f trig=%s",
                        self.serial or "(first)", self._last_size, self.target_fps,
                        "HW" if self.hw_trigger else "Free")
        finally:
            try: lst.Clear()
            except Exception: pass
            try: del lst
            except Exception: pass

    def start_acquisition(self):
        if self.cam and not self._acq:
            try:
                self.cam.BeginAcquisition(); self._acq=True
            except Exception as e:
                msg = str(e).lower()
                if "already streaming" in msg:
                    # Treat as already started to avoid repeated warnings
                    self._acq=True
                else:
                    LOGGER.warning("[PySpin] BeginAcquisition: %s", e)

    def get_frame(self):
        if not self.cam: return None
        if not self._acq: self.start_acquisition()
        try:
            img = self.cam.GetNextImage(50)
            if img.IsIncomplete():
                img.Release(); return None
            arr = img.GetNDArray()
            w = img.GetWidth(); h = img.GetHeight()
            img.Release()
            if arr.ndim==2:
                if HAVE_OPENCV: arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                else: arr = np.repeat(arr[...,None],3,axis=2)
            elif arr.ndim==3 and arr.shape[2]==1:
                arr = np.repeat(arr,3,axis=2)
            self._last_size=(int(w),int(h))
            return arr
        except Exception as e:
            LOGGER.debug("[PySpin] Frame error: %s", e); return None

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
        finally:
            self.cam=None

class CameraNode:
    def __init__(self, name:str, backend:str, ident:str, target_fps:int, adv=None):
        self.name=name; self.backend=backend; self.ident=ident; self.target_fps=float(target_fps)
        self.dev:Optional[BaseCamera]=None; self.synthetic=False; self.preview_times=deque(maxlen=120); self.adv=adv or {}
    def _open_if_needed(self):
        if self.dev is not None or self.synthetic: return
        try:
            if self.backend=="PySpin":
                serial=self.ident
                width=int(self.adv.get("width",0) or 0); height=int(self.adv.get("height",0) or 0)
                exposure_us=int(self.adv.get("exposure_us",1500) or 1500); hw_trig=bool(self.adv.get("hw_trigger",False))
                self.dev=SpinnakerCamera(serial,self.target_fps,width,height,exposure_us,hw_trig); self.dev.open()
                LOGGER.info("[%s] PySpin open: serial=%s %s @ %.1ffps trig=%s", self.name, serial or "(first)", self.dev.frame_size(), self.target_fps, "HW" if hw_trig else "Free")
            else:
                if not HAVE_OPENCV: raise RuntimeError("OpenCV not installed")
                idx=int(self.ident or "0"); dev=OpenCVCamera(idx,self.target_fps); dev.open()
                if getattr(dev, "cap", None) is None: self.synthetic=True; self.dev=None; LOGGER.info("[%s] OpenCV index %d not available → synthetic", self.name, idx)
                else: self.dev=dev; LOGGER.info("[%s] OpenCV open: index %d", self.name, idx)
        except Exception as e:
            LOGGER.warning("[%s] Open error: %s → synthetic", self.name, e); self.dev=None; self.synthetic=True
    def set_backend_ident(self, backend:str, ident:str, adv=None):
        self.release(); self.backend=backend; self.ident=ident; self.synthetic=False
        if adv is not None: self.adv=adv
        LOGGER.info("[%s] set backend=%s ident=%s (lazy open)", self.name, backend, ident)
    def set_target_fps(self,fps:int): self.target_fps=float(fps)
    def grab_preview(self,w:int,h:int):
        self._open_if_needed()
        if self.synthetic or self.dev is None:
            frame=np.full((max(h,1),max(w,1),3),240,dtype=np.uint8)
            if HAVE_OPENCV: cv2.putText(frame,f"{self.name} (synthetic)",(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2,cv2.LINE_AA)
            self.preview_times.append(time.time()); return frame
        img=self.dev.get_frame()
        if img is None:
            frame=np.full((max(h,1),max(w,1),3),255,dtype=np.uint8)
            if HAVE_OPENCV: cv2.putText(frame,f"{self.name} [drop]",(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2,cv2.LINE_AA)
            self.preview_times.append(time.time()); return frame
        if HAVE_OPENCV and (img.shape[1]!=w or img.shape[0]!=h):
            img=cv2.resize(img,(w,h))
        self.preview_times.append(time.time()); return img
    def driver_fps(self)->float:
        if len(self.preview_times)<2: return 0.0
        dt=self.preview_times[-1]-self.preview_times[0]; n=len(self.preview_times)-1
        return (n/dt) if dt>0 else 0.0

    def record_clip(self, out_path:str, duration_s:float, fourcc_str:str, async_writer:bool=True)->str:
        self._open_if_needed()
        if self.synthetic or self.dev is None:
            size=(640,480)
        else:
            size=self.dev.frame_size()
        w,h = size
        ensure_dir(os.path.dirname(out_path) or ".")
        if not HAVE_OPENCV:
            with open(out_path + ".txt", "w", encoding="utf-8") as f:
                f.write(f"{self.name} synthetic recording placeholder\n")
            LOGGER.warning("[%s] OpenCV missing; wrote placeholder text", self.name)
            return out_path + ".txt"
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        _, ext = os.path.splitext(out_path)
        if not ext:
            ext = ".avi" if fourcc_str.upper() in ("MJPG","XVID","DIVX") else ".mp4"
            out_path += ext
        writer = cv2.VideoWriter(out_path, fourcc, max(1.0, float(self.target_fps)), (w,h), True)
        if not writer or not writer.isOpened():
            LOGGER.warning("[%s] Writer open failed for %s fourcc=%s → fallback MJPG/AVI", self.name, out_path, fourcc_str)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            out_path = (out_path.rsplit(".",1)[0] if "." in out_path else out_path)+".avi"
            writer = cv2.VideoWriter(out_path, fourcc, max(1.0, float(self.target_fps)), (w,h), True)
        q = queue.Queue(maxsize=256)
        stop_flag = {"stop": False}
        def wr():
            while True:
                if stop_flag["stop"] and q.empty(): break
                try:
                    frame = q.get(timeout=0.05)
                except queue.Empty:
                    continue
                try: writer.write(frame)
                except Exception as e: LOGGER.warning("[%s] Writer error: %s", self.name, e)
        t_writer = None
        if async_writer:
            t_writer = threading.Thread(target=wr, daemon=True); t_writer.start()
        t0 = time.time()
        if self.dev is not None and not self.synthetic:
            try: self.dev.start_acquisition()
            except Exception: pass
        frames=0; drops=0
        while (time.time()-t0) < float(duration_s):
            if self.synthetic or self.dev is None:
                frame = np.full((h,w,3), 255, np.uint8)
                if HAVE_OPENCV:
                    cv2.putText(frame, f"{self.name} synthetic", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2, cv2.LINE_AA)
            else:
                frame = self.dev.get_frame()
                if frame is None:
                    drops += 1
                    continue
                if frame.shape[1]!=w or frame.shape[0]!=h:
                    try: frame = cv2.resize(frame, (w,h))
                    except Exception: frame = np.full((h,w,3), 0, np.uint8)
            frames += 1
            if async_writer:
                try: q.put_nowait(frame)
                except queue.Full: drops+=1
            else:
                writer.write(frame)
        stop_flag["stop"]=True
        if t_writer: t_writer.join(timeout=2.0)
        try: writer.release()
        except Exception: pass
        if self.dev is not None and not self.synthetic:
            try: self.dev.stop_acquisition()
            except Exception: pass
        LOGGER.info("[%s] Recorded %d frames (drops=%d) → %s", self.name, frames, drops, out_path)
        return out_path

    def probe_max_fps(self, seconds:float=3.0)->Tuple[float,int,int]:
        self._open_if_needed()
        frames=0; drops=0
        t0=time.time()
        try: 
            if self.dev is not None: self.dev.start_acquisition()
        except Exception: pass
        while (time.time()-t0)<seconds:
            img=None if (self.synthetic or self.dev is None) else self.dev.get_frame()
            if img is None: drops+=1
            else: frames+=1
        try: 
            if self.dev is not None: self.dev.stop_acquisition()
        except Exception: pass
        elapsed=max(1e-6, time.time()-t0)
        return (frames/elapsed, frames, drops)

    def release(self):
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
            except Exception as e: LOGGER.warning("[Stim] PsychoPy window error: %s", e); self._pp_win=None
        if self._pp_win is not None:
            try: self._pp_win.color=[bg_grey]*3
            except Exception: pass
    def _cv_window(self,screen_idx:int,bg_grey:float):
        try:
            if not HAVE_OPENCV: return
            if not self._cv_open:
                cv2.namedWindow(self._cv_window_name,cv2.WINDOW_NORMAL); cv2.resizeWindow(self._cv_window_name,self._cv_size[0],self._cv_size[1]); self._cv_open=True
            geoms=QtGui.QGuiApplication.screens()
            if 0<=screen_idx<len(geoms):
                g=geoms[screen_idx].geometry(); cv2.moveWindow(self._cv_window_name,g.x()+50,g.y()+50)
            bg=int(max(0,min(255,int(bg_grey*255)))); frame=np.full((self._cv_size[1],self._cv_size[0],3),bg,dtype=np.uint8)
            cv2.imshow(self._cv_window_name,frame); cv2.waitKey(1)
        except Exception as e: LOGGER.warning("[Stim] OpenCV window error: %s", e); self._cv_open=False
    def open_persistent(self,screen_idx:int,fullscreen:bool,bg_grey:float):
        if _ensure_psychopy_loaded():
            self._pp_window(screen_idx,fullscreen,bg_grey)
            if self._pp_win is not None:
                try: self._pp_win.flip()
                except Exception: pass
        else: self._cv_window(screen_idx,bg_grey)
    def run(self,duration_s:float,r0:int,r1:int,bg_grey:float,screen_idx:int,fullscreen:bool):
        LOGGER.info("[Stim] Looming start")
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
                    LOGGER.info("[Stim] Done (PsychoPy)"); return
            except Exception as e: LOGGER.warning("[Stim] PsychoPy error: %s → OpenCV fallback", e)
        try:
            if not HAVE_OPENCV:
                wait_s(duration_s); LOGGER.info("[Stim] Done (timing only)"); return
            self._cv_window(screen_idx,bg_grey)
            size=self._cv_size; bg=int(max(0,min(255,int(bg_grey*255)))); t0=time.time()
            while True:
                t=time.time()-t0
                if t>=duration_s: break
                r=int(r0+(r1-r0)*(t/duration_s)); frame=np.full((size[1],size[0],3),bg,dtype=np.uint8)
                cv2.circle(frame,(size[0]//2,size[1]//2),r,(0,0,0),-1); cv2.imshow(self._cv_window_name,frame)
                if cv2.waitKey(1) & 0xFF==27: break
            LOGGER.info("[Stim] Done (OpenCV)")
        except Exception as e: LOGGER.warning("[Stim] Fallback display unavailable: %s", e); wait_s(duration_s); LOGGER.info("[Stim] Done (timing only)")
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
        LOGGER.info("[Trial] Recording…")
        c0,c1=self._record_both(folder)
        if self.cfg.lights_delay_s>0: LOGGER.info("[Trial] Wait %.3fs → LIGHTS ON", self.cfg.lights_delay_s); wait_s(self.cfg.lights_delay_s)
        self.hw.lights_on()
        if self.cfg.stim_delay_s>0: LOGGER.info("[Trial] Wait %.3fs → STIM", self.cfg.stim_delay_s); wait_s(self.cfg.stim_delay_s)
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
        LOGGER.info("[Trial] Logged")

# -------------------- Device Enumeration --------------------
def enumerate_opencv(max_index:int=6)->List[str]:
    if not HAVE_OPENCV: return []
    found=[]
    for idx in range(max_index):
        cap=None
        try:
            cap=cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name=="nt" else cv2.CAP_ANY)
            if cap and cap.isOpened():
                found.append(f"OpenCV index {idx}")
        except Exception:
            pass
        finally:
            try:
                if cap: cap.release()
            except Exception: pass
    return found

# -------------------- GUI --------------------
class SettingsGUI(QtWidgets.QWidget):
    start_experiment=QtCore.pyqtSignal(); stop_experiment=QtCore.pyqtSignal(); apply_settings=QtCore.pyqtSignal(); manual_trigger=QtCore.pyqtSignal()
    probe_requested=QtCore.pyqtSignal()
    refresh_devices_requested=QtCore.pyqtSignal()

    def __init__(self,cfg:Config,cam0:CameraNode,cam1:CameraNode):
        super().__init__()
        self.cfg=cfg; self.cam0=cam0; self.cam1=cam1
        self.setWindowTitle(f"FlyPy — v{__version__}")

        outer=QtWidgets.QVBoxLayout(self)
        scroll=QtWidgets.QScrollArea(self); scroll.setWidgetResizable(True)
        pane=QtWidgets.QWidget(); root=QtWidgets.QVBoxLayout(pane)

        # Top row
        row0=QtWidgets.QHBoxLayout()
        row0.addWidget(QtWidgets.QLabel("Quick Preset:"))
        self.cb_preset=QtWidgets.QComboBox()
        self.cb_preset.addItem("Blackfly 522 fps (PySpin Mono8, ROI 640×512)")
        self.cb_preset.addItem("Blackfly 300 fps (PySpin Mono8, ROI 720×540)")
        self.cb_preset.addItem("OpenCV baseline (laptop cam)")
        self.bt_apply_preset=QtWidgets.QPushButton("Apply Preset")
        self.bt_probe=QtWidgets.QPushButton("Probe Max FPS")
        self.bt_refresh=QtWidgets.QPushButton("Refresh Cameras")
        row0.addWidget(self.cb_preset); row0.addWidget(self.bt_apply_preset); row0.addWidget(self.bt_probe); row0.addWidget(self.bt_refresh)
        root.addLayout(row0)
        self.bt_apply_preset.clicked.connect(self._apply_selected_preset)
        self.bt_probe.clicked.connect(self._probe_clicked)
        self.bt_refresh.clicked.connect(self._refresh_clicked)

        # Controls row
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

        self.lbl_status=QtWidgets.QLabel("Status: Idle."); root.addWidget(self.lbl_status)

        grid=QtWidgets.QGridLayout(); root.addLayout(grid)

        # General
        gen=QtWidgets.QGroupBox("General")
        gl=QtWidgets.QFormLayout(gen)
        self.cb_sim=QtWidgets.QCheckBox("Test/Simulation Mode (timer triggers)")
        self.cb_sim.setChecked(self.cfg.simulation_mode); gl.addRow(self.cb_sim)
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
        btn_browse.clicked.connect(self._browse)

        # Stimulus & Timing
        stim=QtWidgets.QGroupBox("Stimulus & Timing (Falling object / growing dot)")
        sl=QtWidgets.QFormLayout(stim)
        self.sb_stim_dur=QtWidgets.QDoubleSpinBox(); self.sb_stim_dur.setRange(0.05,60.0); self.sb_stim_dur.setDecimals(3); self.sb_stim_dur.setValue(self.cfg.stim_duration_s)
        self.sb_r0=QtWidgets.QSpinBox(); self.sb_r0.setRange(1,4000); self.sb_r0.setValue(self.cfg.stim_r0_px)
        self.sb_r1=QtWidgets.QSpinBox(); self.sb_r1.setRange(1,8000); self.sb_r1.setValue(self.cfg.stim_r1_px)
        self.sb_bg=QtWidgets.QDoubleSpinBox(); self.sb_bg.setRange(0.0,1.0); self.sb_bg.setSingleStep(0.05); self.sb_bg.setValue(self.cfg.stim_bg_grey)
        self.sb_light_delay=QtWidgets.QDoubleSpinBox(); self.sb_light_delay.setRange(0.0,10.0); self.sb_light_delay.setDecimals(3); self.sb_light_delay.setValue(self.cfg.lights_delay_s)
        self.sb_stim_delay=QtWidgets.QDoubleSpinBox(); self.sb_stim_delay.setRange(0.0,10.0); self.sb_stim_delay.setDecimals(3); self.sb_stim_delay.setValue(self.cfg.stim_delay_s)
        sl.addRow("Stimulus total time (s):", self.sb_stim_dur)
        sl.addRow("Stimulus Start Size (radius px):", self.sb_r0)
        sl.addRow("Stimulus End Size (radius px):", self.sb_r1)
        sl.addRow("Background shade (0=black, 1=white):", self.sb_bg)
        sl.addRow("Delay: record → lights ON (s):", self.sb_light_delay)
        sl.addRow("Delay: record → stimulus ON (s):", self.sb_stim_delay)
        grid.addWidget(stim, 1, 0, 1, 2)

        # Display & Windows
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

        # Camera panels
        self.cam_boxes=[]
        for idx, node in enumerate((cam0, cam1)):
            gb=QtWidgets.QGroupBox(f"Camera {idx}")
            glb=QtWidgets.QGridLayout(gb)

            preview=QtWidgets.QLabel("Preview OFF"); preview.setFixedSize(360,240)
            preview.setStyleSheet("background:#ddd;border:1px solid #aaa;"); preview.setAlignment(QtCore.Qt.AlignCenter)
            cb_show=QtWidgets.QCheckBox("Show Preview"); cb_show.setChecked(False)
            glb.addWidget(preview,0,0,7,1); glb.addWidget(cb_show,7,0,1,1)

            cb_backend=QtWidgets.QComboBox(); cb_backend.addItem("OpenCV"); cb_backend.addItem("PySpin")
            cb_backend.setCurrentIndex(0 if node.backend=="OpenCV" else 1)
            glb.addWidget(QtWidgets.QLabel("Backend:"),0,1); glb.addWidget(cb_backend,0,2)

            cb_device=QtWidgets.QComboBox(); cb_device.setEditable(False); cb_device.setMinimumWidth(280)
            glb.addWidget(QtWidgets.QLabel("Device:"),1,1); glb.addWidget(cb_device,1,2)

            le_ident=QtWidgets.QLineEdit(node.ident)
            le_ident.setPlaceholderText("OpenCV index (0/1/…) or PySpin serial (e.g., 24102017)")
            glb.addWidget(QtWidgets.QLabel("Manual index/serial:"),2,1); glb.addWidget(le_ident,2,2)

            sb_fps=QtWidgets.QSpinBox(); sb_fps.setRange(1,10000); sb_fps.setValue(int(node.target_fps))
            glb.addWidget(QtWidgets.QLabel("Target FPS:"),3,1); glb.addWidget(sb_fps,3,2)

            adv_frame=QtWidgets.QFrame(); adv_layout=QtWidgets.QFormLayout(adv_frame)
            sb_w=QtWidgets.QSpinBox(); sb_w.setRange(0,20000); sb_w.setSingleStep(2); sb_w.setValue(int(node.adv.get("width",0) or 0))
            sb_h=QtWidgets.QSpinBox(); sb_h.setRange(0,20000); sb_h.setSingleStep(2); sb_h.setValue(int(node.adv.get("height",0) or 0))
            adv_layout.addRow("ROI Width (0=max):", sb_w)
            adv_layout.addRow("ROI Height (0=max):", sb_h)
            sb_exp=QtWidgets.QSpinBox(); sb_exp.setRange(20, 1000000); sb_exp.setSingleStep(50); sb_exp.setValue(int(node.adv.get("exposure_us",1500) or 1500))
            adv_layout.addRow("Exposure (µs):", sb_exp)
            cb_hwtrig=QtWidgets.QCheckBox("Hardware trigger (Line0)"); cb_hwtrig.setChecked(bool(node.adv.get("hw_trigger", False)))
            adv_layout.addRow(cb_hwtrig)
            adv_frame.setVisible(False)
            btn_adv=QtWidgets.QPushButton("Advanced…")
            def _toggle_adv(checked=False, f=adv_frame):
                f.setVisible(not f.isVisible())
                btn_adv.setText("Hide Advanced" if f.isVisible() else "Advanced…")
            btn_adv.clicked.connect(_toggle_adv)
            glb.addWidget(btn_adv,4,1,1,2)
            glb.addWidget(adv_frame,5,1,2,2)

            lbl_rep=QtWidgets.QLabel("Driver-reported FPS: ~0.0"); glb.addWidget(lbl_rep,8,1,1,2)

            self.cam_boxes.append({"gb":gb,"preview":preview,"cb_show":cb_show,"cb_backend":cb_backend,"cb_device":cb_device,
                                   "le_ident":le_ident,"sb_fps":sb_fps,"sb_w":sb_w,"sb_h":sb_h,"sb_exp":sb_exp,"cb_hw":cb_hwtrig,"lbl_rep":lbl_rep})
            grid.addWidget(gb, 3+idx, 0, 1, 2)

        scroll.setWidget(pane); outer.addWidget(scroll)
        self._update_footer=QtWidgets.QLabel("Tips: Use 'Refresh Cameras' after plugging devices. PySpin entries show serial + model. Previews are optional; recordings are always full-res.")
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

    def _probe_clicked(self): self.probe_requested.emit()
    def _refresh_clicked(self): self.refresh_devices_requested.emit()
    def _browse(self):
        d=QtWidgets.QFileDialog.getExistingDirectory(self,"Select output folder", self.le_root.text() or os.getcwd())
        if d: self.le_root.setText(d)

    def set_preview_image(self, cam_idx:int, img_bgr: Optional[np.ndarray]):
        if img_bgr is None:
            self.cam_boxes[cam_idx]["preview"].setText("Preview OFF")
            self.cam_boxes[cam_idx]["preview"].setPixmap(QtGui.QPixmap())
            return
        h,w,_=img_bgr.shape
        if HAVE_OPENCV:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = img_bgr[..., ::-1].copy()
        qimg = QtGui.QImage(rgb.tobytes(), w, h, 3*w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.cam_boxes[cam_idx]["preview"].setPixmap(pix)

    def update_cam_fps_labels(self, f0:float, f1:float):
        self.cam_boxes[0]["lbl_rep"].setText(f"Driver-reported FPS: ~{f0:.1f}")
        self.cam_boxes[1]["lbl_rep"].setText(f"Driver-reported FPS: ~{f1:.1f}")

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
        self.gui.start_experiment.connect(self.start_loop)
        self.gui.stop_experiment.connect(self.stop_loop)
        self.gui.apply_settings.connect(self.apply_from_gui)
        self.gui.manual_trigger.connect(self.trigger_once)
        self.gui.probe_requested.connect(self.start_probe)
        self.gui.refresh_devices_requested.connect(self.refresh_devices)

        self.show_scaled_gui(self.cfg.gui_screen_index)

        self.running=False; self.in_trial=False; self.thread=None
        self.preview_timer=QtCore.QTimer(self); self.preview_timer.setInterval(300); self.preview_timer.timeout.connect(self.update_previews); self.preview_timer.start()

        self.aboutToQuit.connect(self.cleanup)

        if self.cfg.prewarm_stim:
            QtCore.QTimer.singleShot(300, lambda: self.runner.stim.open_persistent(self.cfg.stim_screen_index,self.cfg.stim_fullscreen,self.cfg.stim_bg_grey))

        self._print_startup_summary()
        QtCore.QTimer.singleShot(200, self.refresh_devices)

    def _print_startup_summary(self):
        LOGGER.info("=== FlyPy Startup ===")
        LOGGER.info("Version: %s", __version__)
        LOGGER.info("OpenCV: %s", "OK" if HAVE_OPENCV else "MISSING")
        try:
            import PySpin as _ps; LOGGER.info("PySpin: OK")
        except Exception as e:
            LOGGER.info("PySpin: MISSING (%s) — install Spinnaker SDK + PySpin", e)
        LOGGER.info("======================")

    def show_scaled_gui(self, screen_index:int):
        screens=QtGui.QGuiApplication.screens()
        geo=screens[screen_index].availableGeometry() if 0<=screen_index<len(screens) else QtGui.QGuiApplication.primaryScreen().availableGeometry()
        target_w=max(980,int(geo.width()*0.9)); target_h=max(720,int(geo.height()*0.9))
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

    def refresh_devices(self):
        """Populate device dropdowns and auto-assign distinct defaults."""
        def _find_item_index(cb:QtWidgets.QComboBox, backend:str, ident:str)->int:
            for i in range(cb.count()):
                data = cb.itemData(i)
                if isinstance(data, dict) and data.get("backend")==backend and str(data.get("ident"))==str(ident):
                    return i
            return -1

        try:
            ocv_list = enumerate_opencv(6) if HAVE_OPENCV else []
            spin_list = spin_enumerate()

            # Build a flat ordered list of available devices: PySpin first, then OpenCV
            devs=[]
            for item in spin_list:
                devs.append({"backend":"PySpin","ident":item["serial"],"label":item["display"]})
            for s in ocv_list:
                try: idxnum = int(s.split()[-1])
                except Exception: idxnum = 0
                devs.append({"backend":"OpenCV","ident":str(idxnum),"label":s})

            # Populate each camera's combobox
            for idx, box in enumerate(self.gui.cam_boxes):
                cb = box["cb_device"]
                cb.blockSignals(True)
                cb.clear()
                cb.addItem("— Select a device (or type manually) —")
                cb.model().item(0).setEnabled(False)
                for d in devs:
                    cb.addItem(d["label"], {"backend":d["backend"],"ident":d["ident"]})
                cb.blockSignals(False)

                # Make selecting a device update backend + ident fields
                def on_change(i, box=box, cb=cb):
                    data = cb.itemData(i)
                    if isinstance(data, dict) and "backend" in data:
                        box["cb_backend"].setCurrentIndex(0 if data["backend"]=="OpenCV" else 1)
                        box["le_ident"].setText(str(data["ident"]))
                # Avoid duplicating connections across refreshes:
                try:
                    cb.currentIndexChanged.disconnect()
                except Exception:
                    pass
                cb.currentIndexChanged.connect(on_change)

            # ---- Auto-assign DISTINCT defaults when nothing has been chosen or both are identical ----
            # Current typed/selected idents:
            id0 = self.gui.cam_boxes[0]["le_ident"].text().strip()
            id1 = self.gui.cam_boxes[1]["le_ident"].text().strip()
            be0 = "OpenCV" if self.gui.cam_boxes[0]["cb_backend"].currentIndex()==0 else "PySpin"
            be1 = "OpenCV" if self.gui.cam_boxes[1]["cb_backend"].currentIndex()==0 else "PySpin"

            changed=False
            # Helper to set one box to a specific device
            def _set_box(box, backend, ident):
                nonlocal changed
                box["cb_backend"].setCurrentIndex(0 if backend=="OpenCV" else 1)
                box["le_ident"].setText(str(ident))
                # Try to select matching row in the device combobox (if present)
                i=_find_item_index(box["cb_device"], backend, ident)
                if i>=0: box["cb_device"].setCurrentIndex(i)
                changed=True

            # Case 1: we have at least two devices and panels are blank or duplicate → assign devs[0], devs[1]
            if len(devs) >= 2 and ((not id0 and not id1) or (be0==be1 and id0==id1)):
                _set_box(self.gui.cam_boxes[0], devs[0]["backend"], devs[0]["ident"])
                _set_box(self.gui.cam_boxes[1], devs[1]["backend"], devs[1]["ident"])

            # Case 2: only one device is present and both panels are blank or identical → Cam0 = that device; Cam1 = different index
            elif len(devs) == 1 and ((not id0 and not id1) or (be0==be1 and id0==id1)):
                d0 = devs[0]
                _set_box(self.gui.cam_boxes[0], d0["backend"], d0["ident"])
                # pick a different OpenCV index for cam1 to avoid mirroring (likely becomes synthetic until plugged)
                fallback_ident = "1" if str(d0["ident"])!="1" else "0"
                _set_box(self.gui.cam_boxes[1], "OpenCV", fallback_ident)

            # Case 3: one panel empty → fill it with the first device that does NOT match the other panel
            elif len(devs) >= 1:
                # Refresh latest values
                id0 = self.gui.cam_boxes[0]["le_ident"].text().strip(); be0 = "OpenCV" if self.gui.cam_boxes[0]["cb_backend"].currentIndex()==0 else "PySpin"
                id1 = self.gui.cam_boxes[1]["le_ident"].text().strip(); be1 = "OpenCV" if self.gui.cam_boxes[1]["cb_backend"].currentIndex()==0 else "PySpin"
                if not id0:
                    pick = next((d for d in devs if not (d["backend"]==be1 and str(d["ident"])==id1)), devs[0])
                    _set_box(self.gui.cam_boxes[0], pick["backend"], pick["ident"])
                if not id1:
                    pick = next((d for d in devs if not (d["backend"]==be0 and str(d["ident"])==id0)), devs[0])
                    _set_box(self.gui.cam_boxes[1], pick["backend"], pick["ident"])

            LOGGER.info("[Devices] Refreshed: %d PySpin, %d OpenCV", len(spin_list), len(ocv_list))

            # Apply immediately if we auto-changed anything so previews open the right feeds
            if changed:
                self.apply_from_gui()

        except Exception as e:
            LOGGER.error("[Devices] Refresh error: %s", e)

    def apply_from_gui(self):
        try:
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
            LOGGER.info("[Main] Settings applied")

            if self.cfg.prewarm_stim:
                self.runner.stim.open_persistent(self.cfg.stim_screen_index,self.cfg.stim_fullscreen,self.cfg.stim_bg_grey)
            else:
                try: self.runner.stim.close()
                except Exception: pass
        except Exception as e:
            LOGGER.error("[Main] apply_from_gui error: %s", e)

    def update_previews(self):
        try:
            if self.in_trial:
                self.gui.lbl_status.setText("Status: Trial running (preview paused)")
                self.gui.update_cam_fps_labels(self.cam0.driver_fps(), self.cam1.driver_fps())
                return
            for i, node in enumerate((self.cam0, self.cam1)):
                if not self.gui.cam_boxes[i]["cb_show"].isChecked():
                    self.gui.set_preview_image(i, None)
                    continue
                p = self.gui.cam_boxes[i]["preview"]
                w,h=p.width(),p.height()
                img=node.grab_preview(w,h)
                self.gui.set_preview_image(i,img)
            self.gui.update_cam_fps_labels(self.cam0.driver_fps(), self.cam1.driver_fps())
            self.gui.lbl_status.setText("Status: Waiting / Idle.")
        except Exception as e:
            LOGGER.error("[GUI] update_previews error: %s", e)

    def loop(self):
        LOGGER.info("[Main] Trigger loop started")
        self.gui.lbl_status.setText("Status: Watching for triggers…")
        try:
            while self.running:
                if self.hw.check_trigger():
                    self.in_trial=True
                    self.gui.lbl_status.setText("Status: Triggered — running trial…")
                    try:
                        self.runner.run_one()
                    except Exception as e:
                        LOGGER.error("[Main] Trial error: %s", e)
                    self.in_trial=False
                    self.gui.lbl_status.setText("Status: Waiting / Idle.")
                time.sleep(0.005)
        finally:
            LOGGER.info("[Main] Trigger loop exiting")

    def start_loop(self):
        if self.running: return
        self.running=True
        self.thread=threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        self.gui.lbl_status.setText("Status: Trigger loop running.")
        LOGGER.info("[Main] Start")

    def stop_loop(self):
        if not self.running: return
        self.running=False
        if self.thread:
            self.thread.join(timeout=2.0); self.thread=None
        self.gui.lbl_status.setText("Status: Stopped.")
        LOGGER.info("[Main] Stop")

    def trigger_once(self):
        if self.in_trial: return
        self.in_trial=True
        try:
            self.runner.run_one()
        except Exception as e:
            LOGGER.error("[Main] Manual trial error: %s", e)
        self.in_trial=False

    def start_probe(self):
        try: self.apply_from_gui()
        except Exception as e: LOGGER.error("[Probe] apply_from_gui failed: %s", e)
        self.preview_timer.stop()
        self.gui.lbl_status.setText("Status: Probing max FPS…")
        def worker():
            try:
                res0=self.cam0.probe_max_fps(3.0)
                res1=self.cam1.probe_max_fps(3.0)
                txt=(f"Probe window: 3.0 s\n\n"
                     f"Camera 0 → FPS: {res0[0]:.1f}  (frames={res0[1]}, drops={res0[2]})\n"
                     f"Camera 1 → FPS: {res1[0]:.1f}  (frames={res1[1]}, drops={res1[2]})\n\n"
                     f"Tip: set Target FPS to ~90% of the measured value for stability.")
            except Exception as e:
                txt=f"Probe failed: {e}"
            QtCore.QMetaObject.invokeMethod(self, "_finish_probe", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, txt))
        threading.Thread(target=worker, daemon=True).start()

    @QtCore.pyqtSlot(str)
    def _finish_probe(self, msg:str):
        try:
            self.preview_timer.start()
            self.gui.lbl_status.setText("Status: Probe finished.")
            QtWidgets.QMessageBox.information(self.gui, "Probe Max FPS", msg)
        except Exception as e:
            LOGGER.error("[Probe] finish error: %s", e)

    def cleanup(self):
        LOGGER.info("[Main] Cleanup…")
        try:
            self.preview_timer.stop()
        except Exception: pass
        try:
            self.hw.close()
        except Exception: pass
        for node in (self.cam0, self.cam1):
            try: node.release()
            except Exception: pass
        try: self.runner.stim.close()
        except Exception: pass
        try: _spin_system_release_final()
        except Exception: pass
        LOGGER.info("[Main] Cleanup done")

def main():
    app=MainApp(sys.argv); return app.exec_()

if __name__=="__main__":
    sys.exit(main())
