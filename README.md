# FlyPy — Trigger → Cameras + Lights + Looming Stimulus (All-in-One GUI)

Single-window app that coordinates a trigger (hardware, simulated, or manual) to:

- Record two cameras in sync to disk  
- Turn lights on/off (Elegoo/Arduino over serial, or simulated)  
- Present a looming stimulus (growing black dot) on a chosen monitor  
- Log one CSV row per trial with file paths, timing, FPS, and settings  

Version: the current app version is defined in `FlyAPI.py` as `__version__ = "1.3.0"`.

---

## Change These First (Most Common Setup)

**General**
- **Simulation Mode** — use simulated triggers (no hardware).  
- **Interval between simulated triggers (s)** → `sim_trigger_interval`  
- **Output folder** → `output_root`  
- **Video format / codec** → `video_preset_id` (sets `fourcc`)  
- **Recording duration per trigger (s)** → `record_duration_s`  

**Stimulus & Timing**
- **Stimulus duration (s)** → `stim_duration_s`  
- **Dot radii (px)** → `stim_r0_px`, `stim_r1_px`  
- **Background shade (0=black, 1=white)** → `stim_bg_grey`  
- **Delay from recording start → lights ON (s)** → `lights_delay_s`  
- **Delay from recording start → stimulus ON (s)** → `stim_delay_s`  

**Display & Windows**
- **Stimulus display screen** → `stim_screen_index`  
- **GUI display screen** → `gui_screen_index`  
- **Stimulus fullscreen** → `stim_fullscreen`  
- **Pre-warm stimulus window at launch** (optional; slower startup, faster first trial) → `prewarm_stim`

**Cameras**
- **Camera 0/1 index** → `cam0_index`, `cam1_index`  
- **Target FPS** → `cam0_target_fps`, `cam1_target_fps`

> Delays are **absolute from recording start**. The app ensures events occur at their requested absolute times.

---

## Quick Start

### A) Run with Python (recommended during development)

1. Install Python 3.10 (PsychoPy is only supported on `<3.11`; the app will fall back to OpenCV if PsychoPy is absent).  
2. `pip install -r requirements.txt`  
3. Launch: `python FlyAPI.py`  
   - Optional flags:  
     - `--simulate` → start with Simulation Mode ON  
     - `--prewarm-stim` → open stimulus window during startup

### B) Windows portable build (PyInstaller)

- Use the provided build scripts (see repository) to generate a one-folder distribution.  
- Launch the bundled `FlyPy.exe`.  
- CH340/CH341 drivers are in `dist\FlyPy\temp\drivers\` (or run the driver from the batch helper).

---

## Why startup is fast now

- **Lazy PsychoPy import** — only when stimulus is actually shown.  
- **Lazy camera open** — devices open on first preview/record, not at app launch.  
- **Lazy serial open + MCU settle** — the Elegoo/UNO port opens the first time it’s needed.  
- **No modal “Simulation?” prompt** — Simulation Mode is a checkbox in the GUI (and `--simulate` flag).  
- **Optional “Pre-warm stimulus”** — disabled by default; enables instant first stimulus if you want it.

---

## GUI Reference

**Start** — begin watching for triggers (hardware or simulated)  
**Stop** — stop watching  
**Trigger Once (Manual)** — run one trial now (no hardware needed)  
**Apply Settings** — push current GUI values into the running app  
**Help → Check for Updates…** — check GitHub Releases and stage an update

Window behavior:
- GUI auto-maximizes to your selected **GUI display screen**.
- Stimulus appears on **Stimulus display screen**; windowed mode is draggable; fullscreen pins to that monitor.
- If **Pre-warm stimulus** is enabled, the stimulus window opens at launch; otherwise it opens on first use.

---

## Hardware Notes

- **Elegoo/UNO (CH340)**  
  - On first use, FlyPy auto-detects the serial port. If found, it opens the port and waits ~1.2 s for MCU settle.  
  - Commands used: `MARK START`, `MARK END`, `LIGHT ON`, `LIGHT OFF` (newline-terminated).  
  - If the port cannot be opened, FlyPy automatically switches to Simulation Mode (and logs this).

- **Trigger**  
  - Hardware: device sends a line trigger token when the laser beam is broken (default `0`; legacy `T` also accepted).  
  - Simulation: an internal timer fires based on `sim_trigger_interval`.

---

## Cameras

- Device indices are OpenCV indices (0, 1, …).  
- On Windows, FlyPy prefers DirectShow, then MSMF, then CAP_ANY.  
- If a camera fails to open, FlyPy uses a **synthetic** source (white frame with a moving marker) so workflows remain testable.

---

## Logging

Each trial appends one row to `FlyPy_Output\YYYYMMDD\trials_log.csv`:
timestamp, trial_idx, cam0_path, cam1_path, record_duration_s,
lights_delay_s, stim_delay_s, stim_duration_s,
stim_screen_index, stim_fullscreen, cam0_target_fps, cam1_target_fps,
video_preset_id, fourcc





---

## Requirements

Core: `numpy`, `opencv-python`, `PyQt5`, `pyserial`  
Optional: `psychopy` (installed automatically only on Python `< 3.11`; otherwise the stimulus uses the OpenCV fallback)

See `requirements.txt`.

---

## Troubleshooting

- **No stimulus window?** Ensure `stim_delay_s + stim_duration_s ≤ record_duration_s` and the display screen is connected.  
- **Codec problems?** MP4/`mp4v` is the most broadly compatible. The app falls back to `.mp4` with `mp4v` if your selection fails.  
- **No hardware triggers?** Toggle **Simulation Mode** ON and test; confirm CH340 driver installed for hardware tests.  
- **Cameras swapped?** Change **Camera 0/1 index**; typical values are 0 and 1.

---

## Changelog (since 1.3.x)

- **Startup speedups:** lazy PsychoPy import, lazy camera open, lazy serial open & MCU settle.  
- **UX:** Simulation Mode is a GUI checkbox (no modal prompt); `--simulate` and `--prewarm-stim` CLI flags.  
- **Display:** Optional pre-warm of stimulus window at launch (default OFF).  
- **Windows:** Prefer DirectShow backend before MSMF to reduce device-probe delays.


## Device selection (cameras & displays)

- In the Settings window, choose cameras from the **Camera device** dropdowns. Click **Refresh Cameras** after plugging in/out.
- Choose which monitor shows the **Stimulus** and which shows the **GUI** using the new dropdowns. Click **Refresh Displays** if monitors change.
- The Stimulus can run fullscreen on the chosen screen (toggle in the same section).
