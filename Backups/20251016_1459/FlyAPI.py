import os
import sys, time, csv, atexit, threading, queue
from collections import deque

# Windows-specific environment guards BEFORE importing PyQt5/OpenCV
if os.name == 'nt':
    os.environ.setdefault('QT_QPA_PLATFORM', 'windows')
    os.environ.setdefault('QT_ENABLE_HIGHDPI_SCALING', '1')
    os.environ.setdefault('QT_OPENGL', 'software')
    os.environ.setdefault('OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS', '0')

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QLibraryInfo
plugin_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)
if plugin_path and os.path.isdir(plugin_path):
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
print('[BOOT] PyQt5 imported OK')
print('[BOOT] Qt plugin path =', plugin_path)
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseSoftwareOpenGL, True)
import importlib, numpy as np
from typing import Optional, Tuple, List, Dict
from datetime import datetime
__version__ = "1.10.0"

# Optional libs (import OpenCV after Qt to avoid plugin conflicts)
try:
    import cv2
    HAVE_OPENCV=True
except Exception:
    HAVE_OPENCV=False
    cv2=None  # type: ignore

# PsychoPy lazy
PSY_LOADED=None; visual=None; core=None
def _ensure_psychopy_loaded()->bool:
    global PSY_LOADED, visual, core
    if PSY_LOADED is True: return True
    if PSY_LOADED is False: return False
    try:
        importlib.import_module('psychopy')
    except Exception:
        # Try to locate a standalone PsychoPy install (common on Windows)
        alt = os.environ.get('PSYCHOPY_STANDALONE')
        if alt and os.path.isdir(alt):
            sys.path.insert(0, alt)
        try:
            importlib.import_module('psychopy')
        except Exception:
            visual=None; core=None; PSY_LOADED=False; return False
    try:
        visual=importlib.import_module('psychopy.visual')
        core  =importlib.import_module('psychopy.core')
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
    {"id":"mp4_mp4v","label":"MP4 / mp4v — compatible","fourcc":"mp4v"},
    {"id":"avi_xvid","label":"AVI / XVID — broad compatibility","fourcc":"XVID"},
    {"id":"avi_mjpg","label":"AVI / MJPG — fast writer (large files)","fourcc":"MJPG"},
]
PRESETS_BY_ID={p["id"]:p for p in VIDEO_PRESETS}

class Config:
    def __init__(self):
        self.simulation_mode=False
        self.sim_trigger_interval=5.0
        self.output_root="FlyPy_Output"
        self.prewarm_stim=False
        self.video_preset_id="avi_mjpg"; self.fourcc="MJPG"
        self.record_duration_s=2.0
        self.stim_duration_s=1.5
        self.stim_r0_px=8; self.stim_r1_px=240; self.stim_bg_grey=1.0
        self.lights_delay_s=0.0; self.stim_delay_s=0.0
        self.stim_screen_index=0; self.stim_fullscreen=False; self.gui_screen_index=0
        # Cameras (default PySpin for Blackfly)
        self.cam0_backend="PySpin"; self.cam1_backend="PySpin"
        self.cam0_id=""; self.cam1_id=""
        self.cam0_target_fps=300; self.cam1_target_fps=300
        self.cam0_width=640; self.cam0_height=512
        self.cam1_width=640; self.cam1_height=512
        self.cam0_exposure_us=1500; self.cam1_exposure_us=1500
        self.cam0_hw_trigger=False; self.cam1_hw_trigger=False
        self.cam_async_writer=True

# -------------------- Hardware Bridge (stubbed serial) --------------------
class HardwareBridge:
    def __init__(self,cfg:Config,port:str=None,baud:int=115200):
        self.cfg=cfg; self.simulated=cfg.simulation_mode; self.port=port; self.baud=baud
        self._opened=False; self._last_sim=time.time(); self.ser=None
    def _autodetect_port(self)->Optional[str]:
        try:
            import serial.tools.list_ports
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
                print("[HW] No port found → simulation"); self.simulated=True
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
        backends=[cv2.CAP_ANY]
        if os.name=="nt": backends=[cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
        for be in backends:
            try:
                cap=cv2.VideoCapture(self.index,be)
                if cap and cap.isOpened():
                    try:
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y800"))
                        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                    except Exception: pass
                    try: cap.set(cv2.CAP_PROP_FPS,float(self.target_fps))
                    except Exception: pass
                    self.cap=cap; return
                if cap: cap.release()
            except Exception: pass
        self.cap=None
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
    def release(self):
        if self.cap:
            try: self.cap.release()
            except Exception: pass
            self.cap=None

# ---- PySpin (Spinnaker) backend ----
HAVE_PYSPIN=False; PySpin=None; _SPIN_SYS=None; _SPIN_REF=0
_CLAIMED_SPIN_SERIALS=set()
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
        node.SetIntValue(entry.GetValue()); return True
    except Exception as e:
        print(f"[PySpin] Enum {name}={symbolic} failed: {e}"); return False
def _safe_set_float(nodemap, name, value):
    try:
        node = PySpin.CFloatPtr(nodemap.GetNode(name))
        if not PySpin.IsWritable(node): return False
        lo = node.GetMin(); hi = node.GetMax()
        v = max(lo, min(hi, float(value)))
        node.SetValue(v); return True
    except Exception as e:
        print(f"[PySpin] Float {name}={value} failed: {e}"); return False
def _safe_set_bool(nodemap, name, value:bool):
    try:
        node = PySpin.CBooleanPtr(nodemap.GetNode(name))
        if not PySpin.IsWritable(node): return False
        node.SetValue(bool(value)); return True
    except Exception as e:
        print(f"[PySpin] Bool {name}={value} failed: {e}"); return False
def _align_to_inc(val, inc, lo, hi):
    if inc <= 0: return int(max(lo, min(hi, val)))
    v = int(val // inc * inc)
    return max(int(lo), min(int(hi), int(v)))

def _pyspin_list_devices()->List[Dict[str,str]]:
    out: List[Dict[str,str]] = []
    sys_inst = _spin_system_get()
    if sys_inst is None: return out
    cams = sys_inst.GetCameras()
    try:
        for cam in cams:
            try:
                tl = cam.GetTLDeviceNodeMap()
                serial = None; model=None
                try:
                    sn = PySpin.CStringPtr(tl.GetNode("DeviceSerialNumber"))
                    if sn and PySpin.IsReadable(sn): serial = sn.GetValue()
                except Exception: pass
                try:
                    mn = PySpin.CStringPtr(tl.GetNode("DeviceModelName"))
                    if mn and PySpin.IsReadable(mn): model = mn.GetValue()
                except Exception: pass
                out.append({"serial": serial or "", "model": model or ""})
            except Exception:
                continue
    finally:
        try: cams.Clear()
        except Exception: pass
        _spin_system_release()
    return out

def _opencv_list_devices(max_index: int = 10)->List[Dict[str,str]]:
    out=[]
    if not HAVE_OPENCV: return out
    backends=[cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name=='nt' else [cv2.CAP_ANY]
    for idx in range(max_index+1):
        cap=None
        for be in backends:
            try:
                cap=cv2.VideoCapture(idx, be)
                if cap and cap.isOpened():
                    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    out.append({'index': str(idx), 'label': f'Index {idx} — {w}x{h}'})
                    break
            except Exception:
                pass
            finally:
                if cap: cap.release()
    return out

class SpinnakerCamera(BaseCamera):
    def __init__(self, ident:str, target_fps:float, width:int=0, height:int=0, exposure_us:int=1500, hw_trigger:bool=False):
        ident = (ident or '').strip()
        self.select_index = None
        self.serial = ''
        if ident.lower().startswith('index:'):
            try: self.select_index = int(ident.split(':',1)[1])
            except Exception: self.select_index = None
        elif ident.isdigit() and len(ident) < 6:
            self.select_index = int(ident)
        else:
            self.serial = ident
        self.target_fps=float(target_fps)
        self.req_w=int(width); self.req_h=int(height)
        self.exposure_us=int(exposure_us); self.hw_trigger=bool(hw_trigger)
        self.cam=None; self.node=None; self.snode=None; self.anode=None
        self._acq=False; self._last_size=(640,480); self._mono=True
        self._reported_serial=''

    def open(self):
        sys_inst = _spin_system_get()
        if sys_inst is None:
            raise RuntimeError("PySpin not available; install Spinnaker SDK + PySpin")
        lst = sys_inst.GetCameras()
        try:
            if lst.GetSize()==0:
                raise RuntimeError("No Spinnaker cameras detected")
            chosen=None; chosen_serial=None
            # Try serial first
            if self.serial:
                for cam in lst:
                    sn_val=None
                    try:
                        tl = cam.GetTLDeviceNodeMap()
                        sn = PySpin.CStringPtr(tl.GetNode('DeviceSerialNumber'))
                        if sn and PySpin.IsReadable(sn): sn_val = sn.GetValue()
                    except Exception:
                        sn_val=None
                    if sn_val and self.serial==sn_val:
                        chosen=cam; chosen_serial=sn_val; break
            # Try index
            if chosen is None and self.select_index is not None:
                idx = max(0, min(int(self.select_index), int(lst.GetSize()-1)))
                chosen = lst[idx]
                try:
                    tl = chosen.GetTLDeviceNodeMap()
                    sn = PySpin.CStringPtr(tl.GetNode('DeviceSerialNumber'))
                    if sn and PySpin.IsReadable(sn): chosen_serial = sn.GetValue()
                except Exception: pass
            # Fallback
            if chosen is None:
                chosen = lst[0]
                try:
                    tl = chosen.GetTLDeviceNodeMap()
                    sn = PySpin.CStringPtr(tl.GetNode('DeviceSerialNumber'))
                    if sn and PySpin.IsReadable(sn): chosen_serial = sn.GetValue()
                except Exception: pass
                if self.serial:
                    print(f"[PySpin] Serial {self.serial} not found; using first camera ({chosen_serial or 'unknown'}).")
            self.cam = chosen; self.cam.Init()
            if chosen_serial:
                _CLAIMED_SPIN_SERIALS.add(chosen_serial)
            self.node = self.cam.GetNodeMap(); self.snode = self.cam.GetTLStreamNodeMap(); self.anode = self.cam.GetTLDeviceNodeMap()
            self._reported_serial = chosen_serial or ''

            # Stop acquisition if running
            try:
                stop_cmd = PySpin.CCommandPtr(self.node.GetNode('AcquisitionStop'))
                if stop_cmd and PySpin.IsWritable(stop_cmd): stop_cmd.Execute()
            except Exception: pass

            # Stream buffer handling
            try:
                mode = PySpin.CEnumerationPtr(self.snode.GetNode("StreamBufferCountMode"))
                if PySpin.IsWritable(mode):
                    mode.SetIntValue(mode.GetEntryByName("Manual").GetValue())
                    cnt = PySpin.CIntegerPtr(self.snode.GetNode("StreamBufferCountManual"))
                    if PySpin.IsWritable(cnt): cnt.SetValue(max(int(cnt.GetMin()), min(int(cnt.GetMax()), 64)))
            except Exception as e:
                print(f"[PySpin] Stream buffer note: {e}")
            try:
                _safe_set_enum(self.snode, "StreamBufferHandlingMode", "NewestOnly")
            except Exception: pass

            # Pixel format
            ok = _safe_set_enum(self.node, "PixelFormat", "Mono8")
            self._mono = True if ok else False
            if not ok:
                ok = _safe_set_enum(self.node, "PixelFormat", "BayerRG8")
                self._mono = False if ok else True

            # ROI with offset guards
            ## ROI_SEQUENCE_SAFE
            try:
                w = PySpin.CIntegerPtr(self.node.GetNode("Width"))
                h = PySpin.CIntegerPtr(self.node.GetNode("Height"))
                ox = PySpin.CIntegerPtr(self.node.GetNode("OffsetX"))
                oy = PySpin.CIntegerPtr(self.node.GetNode("OffsetY"))
                maxw = int(w.GetMax()); maxh = int(h.GetMax())
                incw = int(w.GetInc()) or 2; inch = int(h.GetInc()) or 2
                # First, expand to full field if possible (some models require this before shrinking)
                try:
                    if PySpin.IsWritable(w): w.SetValue(maxw)
                    if PySpin.IsWritable(h): h.SetValue(maxh)
                except Exception:
                    pass
                # Compute target ROI
                if self.req_w<=0 or self.req_w>maxw: reqw = maxw
                else: reqw = _align_to_inc(self.req_w, incw, w.GetMin(), maxw)
                if self.req_h<=0 or self.req_h>maxh: reqh = maxh
                else: reqh = _align_to_inc(self.req_h, inch, h.GetMin(), maxh)
                # Center offsets only if adjustable
                try:
                    if ox and PySpin.IsWritable(ox) and int(ox.GetMax())>int(ox.GetMin()):
                        cx = max(0, (maxw - reqw)//(2*incw)*incw)
                        ox.SetValue(cx)
                    elif ox and PySpin.IsWritable(ox):
                        ox.SetValue(0)
                except Exception:
                    pass
                try:
                    if oy and PySpin.IsWritable(oy) and int(oy.GetMax())>int(oy.GetMin()):
                        cy = max(0, (maxh - reqh)//(2*inch)*inch)
                        oy.SetValue(cy)
                    elif oy and PySpin.IsWritable(oy):
                        oy.SetValue(0)
                except Exception:
                    pass
                if PySpin.IsWritable(w): w.SetValue(reqw)
                if PySpin.IsWritable(h): h.SetValue(reqh)
                self._last_size = (int(reqw), int(reqh))
            except Exception as e:
                print(f"[PySpin] ROI note: {e}")

            # Exposure / gain
            _safe_set_enum(self.node, "ExposureAuto", "Off")
            if self.exposure_us>0:
                period_us = 1e6/max(1.0, self.target_fps)
                exp_us = min(self.exposure_us, int(period_us*0.85))
                _safe_set_float(self.node, "ExposureTime", exp_us)
            _safe_set_enum(self.node, "GainAuto", "Off")

            # Frame rate / trigger
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

            # Throughput limit
            try:
                mode = PySpin.CEnumerationPtr(self.node.GetNode('DeviceLinkThroughputLimitMode'))
                if mode and PySpin.IsWritable(mode):
                    try: mode.SetIntValue(mode.GetEntryByName('On').GetValue())
                    except Exception: pass
                dl = PySpin.CFloatPtr(self.node.GetNode('DeviceLinkThroughputLimit'))
                if dl and PySpin.IsWritable(dl):
                    desired_Bps = self._last_size[0]*self._last_size[1]*(1 if self._mono else 1.5)*self.target_fps
                    dl.SetValue(min(dl.GetMax(), max(dl.GetMin(), desired_Bps*1.2)))
            except Exception:
                pass

        finally:
            try: lst.Clear()
            except Exception: pass
            _spin_system_release()

    def start_acquisition(self):
        if self.cam and not self._acq:
            try:
                self.cam.BeginAcquisition(); self._acq=True
            except Exception as e:
                if 'already streaming' in str(e).lower():
                    self._acq=True; print("[PySpin] BeginAcquisition: already streaming → continue")
                else:
                    print(f"[PySpin] BeginAcquisition: {e}")

    def get_frame(self):
        if not self.cam: return None
        if not self._acq: self.start_acquisition()
        try:
            img = self.cam.GetNextImage(15)
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
            msg = str(e)
            print(f"[PySpin] Frame error: {msg}")
            if 'NEW_BUFFER_DATA' in msg or 'GetNextImage' in msg or 'Stream is not started' in msg:
                try:
                    self.stop_acquisition(); time.sleep(0.05); self.start_acquisition()
                except Exception:
                    pass
            return None

    def frame_size(self): return self._last_size
    def stop_acquisition(self):
        if self.cam and self._acq:
            try: self.cam.EndAcquisition()
            except Exception: pass
            self._acq=False
    def release(self):
        try: self.stop_acquisition()
        except Exception: pass
        try:
            if self.cam:
                try: self.cam.DeInit()
                except Exception: pass
                self.cam=None
        finally:
            _spin_system_release()

# -------------------- Camera wrapper with high-FPS grabber --------------------
class CameraNode:
    def __init__(self, name:str, backend:str, ident:str, target_fps:int, adv=None):
        self.name=name; self.backend=backend; self.ident=ident; self.target_fps=float(target_fps)
        self.dev:Optional[BaseCamera]=None; self.synthetic=False; self.adv=adv or {}
        self.preview_times=deque(maxlen=120)
        self._grab_running=False; self._grab_thread_obj=None; self._last_frame=None
        self._cap_times=deque(maxlen=2000); self._lock=threading.Lock()

    def _open_if_needed(self):
        if self.dev is not None or self.synthetic: return
        try:
            if self.backend=="PySpin":
                sys_inst = _spin_system_get()
                if sys_inst is None:
                    print(f"[{self.name}] PySpin not available → synthetic"); self.synthetic=True; return
                width=int(self.adv.get("width",0) or 0); height=int(self.adv.get("height",0) or 0)
                exposure_us=int(self.adv.get("exposure_us",1500) or 1500); hw_trig=bool(self.adv.get("hw_trigger",False))
                self.dev=SpinnakerCamera(self.ident,self.target_fps,width,height,exposure_us,hw_trig); self.dev.open()
                sel_idx = getattr(self.dev, 'select_index', None)
                opened_id = getattr(self.dev,'_reported_serial','') or (f'index:{sel_idx}' if sel_idx is not None else '(first)')
                print(f"[{self.name}] PySpin open: {opened_id}")
            else:
                if not HAVE_OPENCV: raise RuntimeError("OpenCV not installed")
                idx=int(self.ident or "0"); dev=OpenCVCamera(idx,self.target_fps); dev.open()
                if dev.cap is None: self.synthetic=True; self.dev=None; print(f"[{self.name}] OpenCV index {idx} not available → synthetic")
                else: self.dev=dev; print(f"[{self.name}] OpenCV open: index {idx}")
        except Exception as e:
            print(f"[{self.name}] Open error: {e} → synthetic"); self.dev=None; self.synthetic=True

    def set_backend_ident(self, backend:str, ident:str, adv=None):
        self.release(); self.backend=backend; self.ident=ident; self.synthetic=False
        if adv is not None: self.adv=adv
        print(f"[{self.name}] set backend={backend} ident={ident} (lazy open)")

    def set_target_fps(self,fps:int): self.target_fps=float(fps)

    def start_grabber(self):
        if self.synthetic or self.dev is None or self._grab_running: return
        self._grab_running=True
        def loop():
            while self._grab_running:
                try:
                    if self.synthetic or self.dev is None: break
                    img = self.dev.get_frame()
                    if img is not None:
                        with self._lock:
                            self._last_frame = img
                            self._cap_times.append(time.time())
                    else:
                        time.sleep(0.001)
                except Exception:
                    time.sleep(0.001)
        self._grab_thread_obj = threading.Thread(target=loop, daemon=True)
        self._grab_thread_obj.start()

    def stop_grabber(self):
        if not self._grab_running: return
        self._grab_running=False
        try:
            if self._grab_thread_obj: self._grab_thread_obj.join(timeout=1.0)
        except Exception: pass
        self._grab_thread_obj=None

    def capture_fps(self)->float:
        with self._lock:
            if len(self._cap_times) < 2: return 0.0
            dt = self._cap_times[-1] - self._cap_times[0]
            n = len(self._cap_times) - 1
        return (n/dt) if dt>0 else 0.0

    def grab_preview(self,w:int,h:int):
        self._open_if_needed()
        if not self.synthetic and self.dev is not None and not self._grab_running:
            self.start_grabber()
        if self.synthetic or self.dev is None:
            frame=np.full((max(h,1),max(w,1),3),240,dtype=np.uint8)
            if HAVE_OPENCV: cv2.putText(frame,f"{self.name} (synthetic)",(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2,cv2.LINE_AA)
            self.preview_times.append(time.time()); return frame
        with self._lock:
            img = None if self._last_frame is None else self._last_frame.copy()
        if img is None:
            frame=np.full((max(h,1),max(w,1),3),255,dtype=np.uint8)
            if HAVE_OPENCV: cv2.putText(frame,f"{self.name} [drop]",(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2,cv2.LINE_AA)
            self.preview_times.append(time.time()); return frame
        if HAVE_OPENCV: img=cv2.resize(img,(w,h))
        self.preview_times.append(time.time()); return img

    def record_clip(self, dst_path:str, duration_s:float, fourcc:str, async_writer:bool=True)->str:
        self._open_if_needed()
        ensure_dir(os.path.dirname(dst_path) or ".")
        start = time.time()
        fps_target = max(1.0, float(self.target_fps))
        if self.synthetic or self.dev is None:
            print(f"[{self.name}] record_clip: no device")
            return ""
        # Prime size
        img = self.dev.get_frame()
        if img is None:
            print(f"[{self.name}] record_clip: no frames available"); return ""
        size = (int(img.shape[1]), int(img.shape[0]))
        if HAVE_OPENCV:
            out = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*fourcc), fps_target, size, True)
            if not out.isOpened(): print(f"[{self.name}] VideoWriter open failed for {dst_path}"); return ""
        else:
            return ""
        frames=0; drops=0
        while (time.time()-start) < float(duration_s):
            img = self.dev.get_frame()
            if img is None:
                drops += 1; continue
            out.write(img); frames += 1
        try: out.release()
        except Exception: pass
        print(f"[{self.name}] record_clip done: frames={frames} drops={drops} -> {dst_path}")
        return dst_path

    def release(self):
        try: self.stop_grabber()
        except Exception: pass
        try:
            if self.dev and hasattr(self.dev,'_reported_serial') and self.dev._reported_serial:
                try: _CLAIMED_SPIN_SERIALS.discard(self.dev._reported_serial)
                except Exception: pass
            if self.dev: self.dev.release()
        except Exception: pass
        self.dev=None; self.synthetic=False

# -------------------- Stimulus (brief) --------------------
class LoomingStim:
    def __init__(self,cfg:Config):
        self.cfg=cfg; self._pp_win=None; self._pp_cfg=None; self._cv_window_name="Looming Stimulus"; self._cv_open=False; self._cv_size=(800,600)
    def open_persistent(self,screen_idx:int,fullscreen:bool,bg_grey:float):
        if _ensure_psychopy_loaded():
            try:
                if fullscreen: self._pp_win=visual.Window(color=[bg_grey]*3,units='pix',fullscr=True,screen=screen_idx)
                else: self._pp_win=visual.Window(size=self._cv_size,color=[bg_grey]*3,units='pix',fullscr=False,screen=screen_idx,allowGUI=True)
                self._pp_cfg=(screen_idx,fullscreen); self._pp_win.flip()
            except Exception as e: print(f"[Stim] PsychoPy window error: {e}"); self._pp_win=None
    def run(self,duration_s:float,r0:int,r1:int,bg_grey:float,screen_idx:int,fullscreen:bool):
        print("[Stim] Looming start")
        if _ensure_psychopy_loaded() and self._pp_win is not None:
            try:
                dot=visual.Circle(self._pp_win,radius=r0,fillColor='black',lineColor='black'); t0=time.time()
                while True:
                    t=time.time()-t0
                    if t>=duration_s: break
                    r=r0+(r1-r0)*(t/duration_s); dot.radius=r; dot.draw(); self._pp_win.flip()
                print("[Stim] Done (PsychoPy)"); return
            except Exception as e: print(f"[Stim] PsychoPy error: {e}")
        if HAVE_OPENCV:
            try:
                if not self._cv_open:
                    cv2.namedWindow(self._cv_window_name,cv2.WINDOW_NORMAL); cv2.resizeWindow(self._cv_window_name,self._cv_size[0],self._cv_size[1]); self._cv_open=True
                bg=int(max(0,min(255,int(bg_grey*255)))); size=self._cv_size; t0=time.time()
                while True:
                    t=time.time()-t0
                    if t>=duration_s: break
                    r=int(r0+(r1-r0)*(t/duration_s)); frame=np.full((size[1],size[0],3),bg,dtype=np.uint8)
                    cv2.circle(frame,(size[0]//2,size[1]//2),r,(0,0,0),-1); cv2.imshow(self._cv_window_name,frame); cv2.waitKey(1)
                print("[Stim] Done (OpenCV)")
            except Exception as e: print(f"[Stim] Fallback display unavailable: {e}"); wait_s(duration_s)
        else:
            wait_s(duration_s)
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
            try: cam.stop_grabber()
            except Exception: pass
            res[key]=cam.record_clip(pth, float(self.cfg.record_duration_s), fourcc, async_writer=self.cfg.cam_async_writer)
            try: cam.start_grabber()
            except Exception: pass
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
        ]); self.log.flush(); print("[Trial] Logged")

# -------------------- GUI --------------------
class SettingsGUI(QtWidgets.QWidget):
    start_experiment=QtCore.pyqtSignal(); stop_experiment=QtCore.pyqtSignal(); apply_settings=QtCore.pyqtSignal(); manual_trigger=QtCore.pyqtSignal(); probe_requested=QtCore.pyqtSignal()
    def __init__(self,cfg:Config,cam0:CameraNode,cam1:CameraNode):
        super().__init__(); self.cfg=cfg; self.cam0=cam0; self.cam1=cam1
        self.setWindowTitle(f"FlyPy — v{__version__}")
        outer=QtWidgets.QVBoxLayout(self)
        scroll=QtWidgets.QScrollArea(self); scroll.setWidgetResizable(True)
        pane=QtWidgets.QWidget(); root=QtWidgets.QVBoxLayout(pane)

        # Preset row
        row0=QtWidgets.QHBoxLayout()
        row0.addWidget(QtWidgets.QLabel("Quick Preset:"))
        self.cb_preset=QtWidgets.QComboBox()
        self.cb_preset.addItem("Blackfly 300 fps (Mono8, ROI 640×512)")
        self.cb_preset.addItem("OpenCV baseline (dev cam)")
        row0.addWidget(self.cb_preset)
        self.bt_apply_preset=QtWidgets.QPushButton("Apply Preset"); row0.addWidget(self.bt_apply_preset)
        self.bt_probe=QtWidgets.QPushButton("Probe Max FPS"); row0.addWidget(self.bt_probe)
        self.bt_calc=QtWidgets.QPushButton("Preset Calculator…"); row0.addWidget(self.bt_calc)
        self.bt_refresh=QtWidgets.QPushButton("Refresh Cameras"); row0.addWidget(self.bt_refresh)
        root.addLayout(row0)
        self.bt_apply_preset.clicked.connect(self._apply_selected_preset)

        # Controls
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
        self.bt_refresh.clicked.connect(self._refresh_cam_lists)

        self.lbl_status=QtWidgets.QLabel("Status: Idle."); root.addWidget(self.lbl_status)

        grid=QtWidgets.QGridLayout(); root.addLayout(grid)

        # General
        gen=QtWidgets.QGroupBox("General"); gl=QtWidgets.QFormLayout(gen)
        # PsychoPy status + installer
        rowpp = QtWidgets.QHBoxLayout(); self.lbl_pp = QtWidgets.QLabel("PsychoPy: checking…")
        self.btn_pp = QtWidgets.QPushButton("Install PsychoPy…")
        rowpp.addWidget(self.lbl_pp); rowpp.addWidget(self.btn_pp)
        wpp = QtWidgets.QWidget(); wpp.setLayout(rowpp); gl.addRow(wpp)

        self.cb_sim=QtWidgets.QCheckBox("Simulation Mode (timer triggers)"); self.cb_sim.setChecked(self.cfg.simulation_mode); gl.addRow(self.cb_sim)
        self.sb_sim=QtWidgets.QDoubleSpinBox(); self.sb_sim.setRange(0.1,3600.0); self.sb_sim.setDecimals(2); self.sb_sim.setValue(self.cfg.sim_trigger_interval); gl.addRow("Interval between simulated triggers (s):", self.sb_sim)
        self.le_root=QtWidgets.QLineEdit(self.cfg.output_root); btn_browse=QtWidgets.QPushButton("Browse…")
        rowr=QtWidgets.QHBoxLayout(); rowr_w=QtWidgets.QWidget(); rowr_w.setLayout(rowr)
        rowr.addWidget(self.le_root); rowr.addWidget(btn_browse); gl.addRow("Output folder:", rowr_w)
        self.cb_fmt=QtWidgets.QComboBox(); self._id_by_idx={}; current=0
        for i,p in enumerate(VIDEO_PRESETS):
            self.cb_fmt.addItem(p["label"]); self.cb_fmt.setItemData(i,p["id"]); self._id_by_idx[i]=p["id"]
            if p["id"]==self.cfg.video_preset_id: current=i
        self.cb_fmt.setCurrentIndex(current); gl.addRow("Video format / codec:", self.cb_fmt)
        self.sb_rec=QtWidgets.QDoubleSpinBox(); self.sb_rec.setRange(0.1,600.0); self.sb_rec.setDecimals(2); self.sb_rec.setValue(self.cfg.record_duration_s); gl.addRow("Recording duration (s):", self.sb_rec)
        grid.addWidget(gen, 0, 0, 1, 2)
        btn_browse.clicked.connect(self._browse)

        # Hook up PsychoPy installer AFTER widgets exist
        try:
            self.btn_pp.clicked.connect(self._install_psychopy)
            self._update_psychopy_status()
        except Exception as e:
            print('[WARN] PsychoPy status/installer hookup failed:', e)


        # Camera panels
        self.cam_boxes=[]
        for idx, node in enumerate((cam0, cam1)):
            gb=QtWidgets.QGroupBox(f"Camera {idx}")
            glb=QtWidgets.QGridLayout(gb)
            preview=QtWidgets.QLabel(); preview.setFixedSize(360,240); preview.setStyleSheet("background:#ddd;border:1px solid #aaa;"); preview.setAlignment(QtCore.Qt.AlignCenter)
            glb.addWidget(preview,0,0,8,1)

            cb_backend=QtWidgets.QComboBox(); cb_backend.addItem("OpenCV"); cb_backend.addItem("PySpin")
            cb_backend.setCurrentIndex(0 if node.backend=="OpenCV" else 1)
            glb.addWidget(QtWidgets.QLabel("Backend:"),0,1); glb.addWidget(cb_backend,0,2)

            le_ident=QtWidgets.QLineEdit(node.ident); le_ident.setPlaceholderText("OpenCV index (e.g., 0) OR PySpin serial (e.g., 24102007)")
            glb.addWidget(QtWidgets.QLabel("Device index / serial:"),1,1); glb.addWidget(le_ident,1,2)

            # Device dropdown + refresh
            cb_device=QtWidgets.QComboBox(); cb_device.setMinimumWidth(220)
            btn_ref=QtWidgets.QPushButton("↻"); btn_ref.setFixedWidth(28)
            dev_row_w=QtWidgets.QWidget(); dev_row=QtWidgets.QHBoxLayout(dev_row_w); dev_row.setContentsMargins(0,0,0,0)
            dev_row.addWidget(cb_device); dev_row.addWidget(btn_ref)
            glb.addWidget(dev_row_w,1,3)

            btn_pick=QtWidgets.QPushButton("Pick…")
            glb.addWidget(btn_pick,1,4)

            sb_fps=QtWidgets.QSpinBox(); sb_fps.setRange(1,10000); sb_fps.setValue(int(node.target_fps))
            glb.addWidget(QtWidgets.QLabel("Target FPS:"),2,1); glb.addWidget(sb_fps,2,2)

            sb_w=QtWidgets.QSpinBox(); sb_w.setRange(64,10000); sb_w.setSingleStep(2); sb_w.setValue(int(node.adv.get("width",640) or 640))
            sb_h=QtWidgets.QSpinBox(); sb_h.setRange(48,10000); sb_h.setSingleStep(2); sb_h.setValue(int(node.adv.get("height",512) or 512))
            glb.addWidget(QtWidgets.QLabel("ROI Width (0=max):"),3,1); glb.addWidget(sb_w,3,2)
            glb.addWidget(QtWidgets.QLabel("ROI Height (0=max):"),4,1); glb.addWidget(sb_h,4,2)

            sb_exp=QtWidgets.QSpinBox(); sb_exp.setRange(20, 1000000); sb_exp.setSingleStep(50); sb_exp.setValue(int(node.adv.get("exposure_us",1500) or 1500))
            glb.addWidget(QtWidgets.QLabel("Exposure (µs):"),5,1); glb.addWidget(sb_exp,5,2)

            cb_hwtrig=QtWidgets.QCheckBox("Hardware trigger (Line0)"); cb_hwtrig.setChecked(bool(node.adv.get("hw_trigger", False)))
            glb.addWidget(cb_hwtrig,6,1,1,2)

            lbl_rep=QtWidgets.QLabel("Capture FPS: ~0.0"); glb.addWidget(lbl_rep,7,1,1,4)

            self.cam_boxes.append({"gb":gb,"preview":preview,"cb_backend":cb_backend,"le_ident":le_ident,"sb_fps":sb_fps,
                                   "sb_w":sb_w,"sb_h":sb_h,"sb_exp":sb_exp,"cb_hw":cb_hwtrig,"lbl_rep":lbl_rep,
                                   "btn_pick":btn_pick,"cb_device":cb_device,"btn_ref":btn_ref})
            grid.addWidget(gb, 1+idx, 0, 1, 2)

        # Display & Stimulus
        disp=QtWidgets.QGroupBox("Display & Stimulus"); dl=QtWidgets.QFormLayout(disp)
        self.cb_stim_screen=QtWidgets.QComboBox(); self.cb_gui_screen=QtWidgets.QComboBox()
        for i,s in enumerate(QtGui.QGuiApplication.screens()):
            g=s.geometry(); label=f"Screen {i} — {g.width()}×{g.height()} @({g.x()},{g.y()})"
            self.cb_stim_screen.addItem(label); self.cb_gui_screen.addItem(label)
        self.cb_stim_screen.setCurrentIndex(self.cfg.stim_screen_index); self.cb_gui_screen.setCurrentIndex(self.cfg.gui_screen_index)
        self.cb_full=QtWidgets.QCheckBox("Stimulus fullscreen"); self.cb_full.setChecked(self.cfg.stim_fullscreen)
        self.cb_prewarm=QtWidgets.QCheckBox("Pre-warm stimulus window at launch"); self.cb_prewarm.setChecked(self.cfg.prewarm_stim)
        dl.addRow("Stimulus display screen:", self.cb_stim_screen); dl.addRow("GUI display screen:", self.cb_gui_screen)
        dl.addRow(self.cb_full); dl.addRow(self.cb_prewarm)
        grid.addWidget(disp, 3, 0, 1, 2)

        scroll.setWidget(pane); outer.addWidget(scroll)

        # Caches
        self._refresh_pyspin_cache: List[Dict[str,str]] = []
        self._refresh_opencv_cache: List[Dict[str,str]] = []

        # Wire buttons
        for i,box in enumerate(self.cam_boxes):
            box["btn_pick"].clicked.connect(lambda _=None, idx=i: self._pick_cam(idx))
            box["cb_backend"].currentIndexChanged.connect(lambda _, idx=i: self._populate_device_list(idx))
            box["btn_ref"].clicked.connect(lambda _, idx=i: self._populate_device_list(idx, force_refresh=True))
            box["cb_device"].currentIndexChanged.connect(lambda _, idx=i: self._apply_device_choice(idx))
            box["cb_device"].addItem('(click ↻ to refresh)')

    def _update_psychopy_status(self):
        # probe within current interpreter
        try:
            import importlib; importlib.import_module('psychopy')
            self.lbl_pp.setText('PsychoPy: available')
            self.btn_pp.setEnabled(False)
        except Exception:
            self.lbl_pp.setText('PsychoPy: NOT installed in current venv')
            self.btn_pp.setEnabled(True)

    def _install_psychopy(self):
        # Inline pip install into this venv; block UI with simple dialog
        dlg = QtWidgets.QMessageBox(self)
        dlg.setWindowTitle('Install PsychoPy')
        dlg.setText('This will install PsychoPy into the current virtual environment. Continue?')
        dlg.setStandardButtons(QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok)
        if dlg.exec_() != QtWidgets.QMessageBox.Ok:
            return
        self.btn_pp.setEnabled(False); self.lbl_pp.setText('PsychoPy: installing… (this may take several minutes)')
        QtWidgets.QApplication.processEvents()
        import subprocess, sys
        try:
            # Use a conservative version known to be stable on Windows
            cmd = [sys.executable, '-m', 'pip', 'install', 'psychopy==2023.2.3']
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            out = proc.stdout[-800:]
            if proc.returncode == 0:
                self.lbl_pp.setText('PsychoPy: installed. Please restart FlyPy to use PsychoPy renderer.')
            else:
                self.lbl_pp.setText('PsychoPy install failed (see console). Still using OpenCV fallback.')
                print(out)
        except Exception as e:
            self.lbl_pp.setText(f'PsychoPy install failed: {e}')
        finally:
            self.btn_pp.setEnabled(True)

    def _apply_selected_preset(self):
        idx=self.cb_preset.currentIndex()
        # Prefer MJPG for high-FPS
        for i in range(self.cb_fmt.count()):
            if (self.cb_fmt.itemData(i) or "").lower()=="avi_mjpg": self.cb_fmt.setCurrentIndex(i); break
        if idx==0:
            for box in self.cam_boxes:
                box["cb_backend"].setCurrentIndex(1)  # PySpin
                box["sb_fps"].setValue(300)
                box["sb_w"].setValue(640); box["sb_h"].setValue(512)
                box["sb_exp"].setValue(1500); box["cb_hw"].setChecked(False)
        else:
            for n,box in enumerate(self.cam_boxes):
                box["cb_backend"].setCurrentIndex(0)  # OpenCV
                box["le_ident"].setText(str(n))
                box["sb_fps"].setValue(60)
                box["sb_w"].setValue(640); box["sb_h"].setValue(480)
                box["sb_exp"].setValue(5000); box["cb_hw"].setChecked(False)
        self.apply_settings.emit()

    def _probe_clicked(self):
        self.probe_requested.emit()

    def _browse(self):
        d=QtWidgets.QFileDialog.getExistingDirectory(self,"Select output folder", self.le_root.text() or os.getcwd())
        if d: self.le_root.setText(d)

    def _refresh_cam_lists(self):
        try: self._refresh_pyspin_cache = _pyspin_list_devices()
        except Exception: self._refresh_pyspin_cache = []

    def _pick_cam(self, cam_idx:int):
        if not self._refresh_pyspin_cache:
            self._refresh_cam_lists()
        lst = self._refresh_pyspin_cache
        if not lst:
            QtWidgets.QMessageBox.warning(self, "No PySpin cameras", "No cameras found via Spinnaker.\nEnsure SDK is installed and cameras are connected.")
            return
        dlg = QtWidgets.QDialog(self); dlg.setWindowTitle(f"Select Camera {cam_idx}")
        lay = QtWidgets.QVBoxLayout(dlg); view = QtWidgets.QListWidget()
        for d in lst: view.addItem(f"{d.get('model','?')}  SN:{d.get('serial','?')}")
        lay.addWidget(view)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel); lay.addWidget(btns)
        btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject)
        if dlg.exec_()==QtWidgets.QDialog.Accepted and view.currentRow()>=0:
            sel = lst[view.currentRow()]
            box = self.cam_boxes[cam_idx]
            box["cb_backend"].setCurrentIndex(1)  # PySpin
            box["le_ident"].setText(sel.get("serial",""))
            QtWidgets.QMessageBox.information(self, "Camera Selected", f"Camera {cam_idx} set to serial: {sel.get('serial','')}")

    def _populate_device_list(self, cam_idx:int, force_refresh:bool=False):
        box = self.cam_boxes[cam_idx]
        be = "OpenCV" if box["cb_backend"].currentIndex()==0 else "PySpin"
        if be=="PySpin":
            if force_refresh or not self._refresh_pyspin_cache: self._refresh_cam_lists()
            items = [f"{d.get('model','?')}  SN:{d.get('serial','?')}" for d in self._refresh_pyspin_cache]
        else:
            if force_refresh or not self._refresh_opencv_cache: self._refresh_opencv_cache = _opencv_list_devices(10)
            items = [d.get('label', f"Index {d.get('index','?')}") for d in self._refresh_opencv_cache]
        cb = box["cb_device"]; cb.blockSignals(True); cb.clear()
        if not items: cb.addItem("(none found)")
        else: cb.addItems(items)
        cb.blockSignals(False)

    def _apply_device_choice(self, cam_idx:int):
        box = self.cam_boxes[cam_idx]
        be = "OpenCV" if box["cb_backend"].currentIndex()==0 else "PySpin"
        idx = box["cb_device"].currentIndex()
        if be=="PySpin":
            if 0 <= idx < len(self._refresh_pyspin_cache):
                serial = self._refresh_pyspin_cache[idx].get("serial","")
                box["le_ident"].setText(serial)
        else:
            if 0 <= idx < len(self._refresh_opencv_cache):
                index = self._refresh_opencv_cache[idx].get("index","0")
                box["le_ident"].setText(index)

    def set_preview_image(self, cam_idx:int, img_rgb: np.ndarray):
        if img_rgb is None: return
        h,w,_=img_rgb.shape
        qimg=QtGui.QImage(img_rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        pix=QtGui.QPixmap.fromImage(qimg)
        self.cam_boxes[cam_idx]["preview"].setPixmap(pix)

    def update_cam_fps_labels(self, f0:float, f1:float):
        self.cam_boxes[0]["lbl_rep"].setText(f"Capture FPS: ~{f0:.1f}")
        self.cam_boxes[1]["lbl_rep"].setText(f"Capture FPS: ~{f1:.1f}")

# -------------------- Main App --------------------
class MainApp(QtWidgets.QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.cfg=Config()
        self.hw=HardwareBridge(self.cfg)
        self.cam0=CameraNode("cam0", self.cfg.cam0_backend, self.cfg.cam0_id, self.cfg.cam0_target_fps,
                             adv={"width":self.cfg.cam0_width,"height":self.cfg.cam0_height,"exposure_us":self.cfg.cam0_exposure_us,"hw_trigger":self.cfg.cam0_hw_trigger})
        self.cam1=CameraNode("cam1", self.cfg.cam1_backend, self.cfg.cam1_id, self.cfg.cam1_target_fps,
                             adv={"width":self.cfg.cam1_width,"height":self.cfg.cam1_height,"exposure_us":self.cfg.cam1_exposure_us,"hw_trigger":self.cfg.cam1_hw_trigger})
        ensure_dir(self.cfg.output_root); log_path=os.path.join(self.cfg.output_root,"trials_log.csv")
        self.runner=TrialRunner(self.cfg,self.hw,self.cam0,self.cam1,log_path)
        self.gui=SettingsGUI(self.cfg,self.cam0,self.cam1)
        # AUTO_UNIQUE_AT_START: ensure two distinct PySpin serials if available
        try:
            lst = _pyspin_list_devices()
            ser = [d.get('serial','') for d in lst if d.get('serial','')]
            if len(ser) >= 2:
                if self.cam0.backend=='PySpin' and self.cam1.backend=='PySpin':
                    if (not self.cam0.ident and not self.cam1.ident) or (self.cam0.ident == self.cam1.ident):
                        self.cam0.ident = ser[0]
                        self.cam1.ident = ser[1]
                        print(f"[Main] Auto-assigned unique serials at startup: cam0={ser[0]} cam1={ser[1]}")
        except Exception:
            pass

        self.gui.start_experiment.connect(self.start_loop)
        self.gui.stop_experiment.connect(self.stop_loop)
        self.gui.apply_settings.connect(self.apply_from_gui)
        self.gui.manual_trigger.connect(self.trigger_once)
        self.gui.probe_requested.connect(self.start_probe)

        self.running=False; self.in_trial=False; self.thread=None
        self.preview_timer=QtCore.QTimer(self); self.preview_timer.setInterval(60); self.preview_timer.timeout.connect(self.update_previews); self.preview_timer.start()

        self.aboutToQuit.connect(self.cleanup); atexit.register(self.cleanup)

        print("=== FlyPy Startup ===")
        print(f"Version: {__version__}")
        print(f"OpenCV: {'OK' if HAVE_OPENCV else 'MISSING'}")
        import importlib.util as _ilu
        ps_spec = _ilu.find_spec('PySpin')
        print("PySpin: OK" if ps_spec is not None else "PySpin: not found")
        print(f"PsychoPy: {'will use' if _ensure_psychopy_loaded() else 'not available (OpenCV fallback)'}")
        print("======================")

    def apply_from_gui(self):
        prev_sim=self.cfg.simulation_mode
        self.cfg.simulation_mode=bool(self.gui.cb_sim.isChecked())
        self.cfg.sim_trigger_interval=float(self.gui.sb_sim.value())
        self.cfg.output_root=self.gui.le_root.text().strip() or self.cfg.output_root

        idx=self.gui.cb_fmt.currentIndex(); preset_id=self.gui.cb_fmt.itemData(idx) or "avi_mjpg"
        self.cfg.video_preset_id=preset_id; self.cfg.fourcc=PRESETS_BY_ID[preset_id]["fourcc"]
        self.cfg.record_duration_s=float(self.gui.sb_rec.value())

        self.cfg.stim_screen_index=int(self.gui.cb_stim_screen.currentIndex())
        self.cfg.gui_screen_index=int(self.gui.cb_gui_screen.currentIndex())
        self.cfg.stim_fullscreen=bool(self.gui.cb_full.isChecked())
        self.cfg.prewarm_stim=bool(self.gui.cb_prewarm.isChecked())

        for i,node in enumerate((self.cam0,self.cam1)):
            box=self.gui.cam_boxes[i]
            backend="OpenCV" if box["cb_backend"].currentIndex()==0 else "PySpin"
            ident=box["le_ident"].text().strip()
            fps=int(box["sb_fps"].value())
            adv={
                "width": int(box["sb_w"].value() or 0),
                "height": int(box["sb_h"].value() or 0),
                "exposure_us": int(box["sb_exp"].value() or 0),
                "hw_trigger": bool(box["cb_hw"].isChecked())
            }
            if backend=="OpenCV" and not ident: ident = str(i)
            if backend=="PySpin" and not ident:
                # Try to auto-assign i-th enumerated camera
                try:
                    lst=_pyspin_list_devices()
                    if 0<=i<len(lst): ident=lst[i].get("serial","")
                except Exception: pass
            node.set_backend_ident(backend, ident, adv=adv)
            node.set_target_fps(fps)
        ## UNIQUE_PYSPIN_ASSIGN
        # Enforce distinct PySpin serials if two are available
        try:
            py_list = _pyspin_list_devices()
        except Exception:
            py_list = []
        serials = [d.get("serial","") for d in py_list if d.get("serial","")]
        cam_idents = [self.cam0.ident if self.cam0.backend=='PySpin' else None,
                      self.cam1.ident if self.cam1.backend=='PySpin' else None]
        # If both cam idents are blank or equal and we have >=2 cameras, assign distinct serials
        if self.cam0.backend=='PySpin' and self.cam1.backend=='PySpin' and len(serials) >= 2:
            if not cam_idents[0] and not cam_idents[1]:
                self.cam0.ident = serials[0]
                self.cam1.ident = serials[1]
                print(f"[Main] Auto-assigned PySpin serials: cam0={serials[0]} cam1={serials[1]}")
            elif cam_idents[0] == cam_idents[1]:
                # Choose a different one for cam1 if possible
                for s in serials:
                    if s != self.cam0.ident:
                        self.cam1.ident = s
                        print(f"[Main] Adjusted cam1 to different serial: {s}")
                        break
        
        ensure_dir(self.cfg.output_root)
        self.gui.lbl_status.setText("Status: Settings applied.")
        print("[Main] Settings applied")

    def update_previews(self):
        if self.in_trial:
            self.gui.lbl_status.setText("Status: Trial running (preview paused)")
            self.gui.update_cam_fps_labels(self.cam0.capture_fps(), self.cam1.capture_fps())
            return
        p0=self.gui.cam_boxes[0]["preview"]; p1=self.gui.cam_boxes[1]["preview"]
        w0,h0=p0.width(),p0.height(); w1,h1=p1.width(),p1.height()
        img0=self.cam0.grab_preview(w0,h0); img1=self.cam1.grab_preview(w1,h1)
        self.gui.set_preview_image(0,img0); self.gui.set_preview_image(1,img1)
        self.gui.update_cam_fps_labels(self.cam0.capture_fps(), self.cam1.capture_fps())
        self.gui.lbl_status.setText("Status: Waiting / Idle.")

    def loop(self):
        self.gui.lbl_status.setText("Status: Watching for triggers…"); print("[Main] Trigger loop started")
        try:
            while self.running:
                if self.hw.check_trigger():
                    self.in_trial=True
                    self.gui.lbl_status.setText("Status: Triggered — running trial…")
                    try: self.runner.run_one()
                    except Exception as e: print(f"[Main] Trial error: {e}")
                    self.in_trial=False; self.gui.lbl_status.setText("Status: Waiting / Idle.")
                QtWidgets.QApplication.processEvents(); time.sleep(0.005)
        finally:
            print("[Main] Trigger loop exiting")

    def start_loop(self):
        if self.running: return
        self.running=True
        self.thread=threading.Thread(target=self.loop, daemon=True); self.thread.start()
        self.gui.lbl_status.setText("Status: Trigger loop running."); print("[Main] Start")

    def stop_loop(self):
        if not self.running: return
        self.running=False
        if self.thread: self.thread.join(); self.thread=None
        self.gui.lbl_status.setText("Status: Stopped."); print("[Main] Stop")

    def trigger_once(self):
        if self.in_trial: return
        self.in_trial=True
        try: self.runner.run_one()
        except Exception as e: print(f"[Main] Manual trial error: {e}")
        self.in_trial=False

    def start_probe(self):
        # Quick capture-rate probe using grabbers for 3s
        self.update_previews()  # ensure opened
        for cam in (self.cam0, self.cam1):
            try: cam.start_grabber()
            except Exception: pass
        t0=time.time(); time.sleep(3.0)
        f0=self.cam0.capture_fps(); f1=self.cam1.capture_fps()
        QtWidgets.QMessageBox.information(self.gui, "Probe Max FPS", f"3s window:\nCam0 ≈ {f0:.1f} fps\nCam1 ≈ {f1:.1f} fps")

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
    try:
        app=MainApp(sys.argv)
    except Exception as e:
        print('[FATAL] QApplication/MainApp init failed:', e)
        return 1
    try:
        rc = app.exec_()
    except Exception as e:
        print('[FATAL] Qt event loop failed:', e)
        rc = 1
    sys.exit(rc)
if __name__=="__main__":
    print('[BOOT] Launching FlyPy main()')
    main()