# FlyPy — Trigger → Cameras + Lights + Looming Stimulus (All-in-One GUI)

Single-window app that coordinates a trigger (hardware, simulated, or manual) to:

- Record two cameras in sync to disk  
- Turn lights on/off (Elegoo/Arduino over serial, or simulated)  
- Present a looming stimulus (growing black dot) on a chosen monitor  
- Log one CSV row per trial with file paths, timing, FPS, and settings  

Version: the current app version is defined in `FlyAPI.py` as `version = "1.3.0"`.

---

# 1) Quick Start (Do This First)

## A. Run with Python installed (fastest path)

1) Open a terminal in the project folder and run:  
```bash
python FlyAPI.py
```

2) On launch, you’ll be asked: **Simulation Mode?**  
- **Yes** → The app “triggers” itself on a timer (`sim_trigger_interval`).  
- **No** → Use hardware: Elegoo/Arduino sends `T` over serial when your sensor/button fires.

3) In the GUI top bar:
- Click **Start** to begin watching for triggers.  
- Click **Trigger Once (Manual)** to run a single trial immediately (no hardware needed).  
- Click **Stop** to end trigger watching.

4) Find your outputs in your chosen **Output folder** (`output_root`):  
- Videos: `_trial####_cam0.*` and `_trial####_cam1.*`  
- Log: `trials_log.csv`

## B. Windows portable build (no Python required)

1) Run the build script:
```
build_flypy_full.bat
```

2) Launch the app:
```
dist\FlyPy\FlyPy.exe
```

3) If Windows doesn’t detect your Elegoo board, install the driver:
```
dist\FlyPy\temp\drivers\CH341SER.EXE
```
(or run `dist\FlyPy\temp\scripts\install_drivers.bat`)

---

# 2) Change These Variables/Controls First (Most Common Setup)

Use the GUI fields (plain-English labels) to change runtime values. The names in **code** are shown for reference.

## General
- **Interval between simulated triggers (seconds)** → `sim_trigger_interval`  
  *Use when Simulation Mode = Yes.*
- **Output folder for all trials** → `output_root`  
  *Where videos and CSV log are saved.*
- **Video file format / codec** → `video_preset_id`  
  *Automatically sets `fourcc`; if a codec/container fails locally, FlyPy falls back to `.mp4` with `mp4v`.*
- **Recording duration per trigger (seconds)** → `record_duration_s`

## Stimulus & Timing
- **Stimulus display duration (seconds)** → `stim_duration_s`
- **Starting / Final dot radius (px)** → `stim_r0_px`, `stim_r1_px`
- **Background shade (0=black, 1=white)** → `stim_bg_grey`
- **Delay from recording start → lights ON (seconds)** → `lights_delay_s`
- **Delay from recording start → stimulus ON (seconds)** → `stim_delay_s`

## Display & Windows
- **Stimulus display screen** → `stim_screen_index`
- **GUI display screen** → `gui_screen_index`
- **Stimulus fullscreen** → `stim_fullscreen` (checked = fullscreen on the selected monitor)

## Cameras (set per camera)
- **OpenCV device index** → `cam0_index`, `cam1_index`
- **Target recording FPS** → `camN_target_fps`  
  *Labels show driver-reported FPS, measured preview FPS, and target FPS for diagnostics.*

After editing values, click **Apply Settings** to push them into the running app.

---

# 3) Timing Model (What Happens Each Trial)

1) **Time 0**: Both cameras start recording.  
2) **At `lights_delay_s`**: Lights turn **ON**.  
3) **At `stim_delay_s`**: Looming stimulus turns **ON**, runs for `stim_duration_s`.  
4) **End of recording**: Cameras stop; lights **OFF**; CSV row appended.

Notes:
- Delays are absolute from the same origin (recording start).  
- The app staggers waits so each event occurs at its requested **absolute** time even if events overlap.

---

# 4) GUI Reference (What the Buttons Do)

**Start** — Begin watching for triggers (hardware or simulated)  
**Stop** — Stop watching  
**Trigger Once (Manual)** — Run one trial immediately (no hardware needed)  
**Apply Settings** — Push the current GUI values into the running app  
**Help → Check for Updates…** — Query GitHub Releases, download the newest portable build, and stage a self-update

Window behavior:
- **Auto-scaled GUI**: main window maximizes to the selected GUI screen’s available size (no fixed 1920×1080).  
- **Persistent stimulus window**: created on app start and remains until quit.  
  - Windowed mode: standard OS window you can drag between monitors (Windows).  
  - Fullscreen: pins to the selected monitor.

---

# 5) Installation & Launch Details

## With Python installed
- Requirements: see `requirements.txt` (core: `numpy`, `opencv-python`, `PyQt5`, `pyserial`; PsychoPy is optional at runtime).  
- Launch: `python FlyAPI.py`

## Windows portable (PyInstaller)
- **Spec**: `FlyPy_full.spec` collects Qt, OpenCV, PsychoPy assets.  
- **Build script**: `build_flypy_full.bat` produces a one-folder “FlyPy-Full” at `dist\FlyPy\`.  
- **Driver**: CH340/CH341 USB-serial driver at `dist\FlyPy\temp\drivers\CH341SER.EXE`.

---

# 6) Hardware Overview (Elegoo/Arduino)

**Trigger input**  
Your sensor (laser break, pushbutton, etc.) toggles a microcontroller pin; the board sends a line with `T` over serial on each trigger.

**Serial protocol (already supported)**  
- **Host → Board**: `START`, `STIM`, `END`, `PULSE <ms>`, `LIGHT ON`, `LIGHT OFF`  
- **Board → Host**: `T` (one line per trigger)

**Firmware**  
Use `ElegooFlyPySync.ino` as-is; it matches the protocol above.

---

# 7) Outputs & Logging

**Videos**  
- `"<output_root>\\<timestamp>_trial####_cam0.<ext>"`  
- `"<output_root>\\<timestamp>_trial####_cam1.<ext>"`  
*(Exact naming may include timestamp/prefix; trial number always increments.)*

**CSV log** → `<output_root>/trials_log.csv` (columns):
```
trial, timestamp, cam0_path, cam1_path, record_duration_s,
lights_delay_s, stim_delay_s, stim_duration_s, stim_screen_index,
stim_fullscreen, cam0_target_fps, cam1_target_fps, video_preset_id, fourcc
```

**Codec fallback**  
If the requested FOURCC/container isn’t supported on the local system, FlyPy auto-falls back to `.mp4 (mp4v)` for compatibility.

---

# 8) Auto-Update (Portable Builds)

**User flow**
- Menu: **Help → Check for Updates…**  
- If a newer GitHub Release with the expected asset (e.g., `FlyPy-Full.zip`) is found, the app downloads it to a temp folder and prepares `update_on_restart.bat` inside the app directory.  
- Close the app (or click a **Close & Update** button if you wire one) to apply the update and relaunch.

**Developer setup**
1) Publish portable builds as Release assets (e.g., `FlyPy-Full.zip`).  
2) In `FlyAPI.py` → `MainApp.on_check_updates`, set:
   - `REPO = "your-github-user/your-repo"`  
   - `ASSET_NAME = "FlyPy-Full.zip"`
3) Keep `version` in `FlyAPI.py` up-to-date so comparisons work.  
4) Updater implementation lives in `auto_update.py` (stdlib only).

---

# 9) Tips, Diagnostics & Troubleshooting

- **No hardware?** Use **Simulation Mode = Yes** or **Trigger Once (Manual)**.  
- **No PsychoPy installed?** The looming stimulus uses an OpenCV window (already implemented).  
- **Wrong monitor?** Set **Stimulus display screen**; if windowed, drag the stimulus. The app remembers the window between trials.  
- **Strange FPS or codec issues?** Prefer MP4 (`mp4v`) for broad compatibility.  
- **Serial device missing?** Install CH340/CH341 driver from the portable build’s `temp\drivers\` folder.  
- **Desync suspicion?** Check the CSV log for `record_duration_s`, delays, and measured FPS to confirm exact timings.  
- **Camera indexing**: If a camera doesn’t open, try swapping `cam0_index`/`cam1_index` (typical values are `0`, `1`, etc.).  
- **Lighting didn’t fire?** Verify `lights_delay_s` > 0 and total recording duration is long enough for the light event to occur before stop.  
- **Stimulus never appeared?** Confirm `stim_delay_s` + `stim_duration_s` ≤ `record_duration_s`, and that the stimulus screen is connected/enabled.

---

# 10) File Map

- `FlyAPI.py` — Main application (GUI, cameras, stimulus, hardware, logging, updater hook)  
- `auto_update.py` — GitHub Releases checker & Windows-safe self-update stager  
- `ElegooFlyPySync.ino` — Arduino sketch (trigger + light/sync protocol)  
- `FlyPy_full.spec` — PyInstaller spec for “FlyPy-Full” build (includes PsychoPy assets)  
- `build_flypy_full.bat` — One-folder build script (creates `dist\FlyPy\`)  
- `requirements.txt` — Core dependencies (PsychoPy optional at runtime)  
- `LICENSE`, `README.md` — Docs & license

---

# 11) Advanced Notes

- **Auto-scaled GUI**: Uses the selected GUI screen’s available size, avoiding fixed 1920×1080 layouts and improving readability on high-DPI/varied displays.  
- **Display routing**: Independent dropdowns for GUI vs. Stimulus screens, plus optional fullscreen for the stimulus.  
- **Independent delays**: `lights_delay_s` and `stim_delay_s` are absolute from recording start; you can fire lights and stimulus in either order or at the same time.  
- **Persistent stimulus window**: Created at app start; avoids window creation latency per trial; draggable when not in fullscreen.

---

# 12) License

See `LICENSE`.

---

# 13) Changelog (since 1.3.x)

- **Auto-update from GitHub**  
  Help → Check for Updates… downloads the latest portable build from your Releases and stages it for installation after exit.

- **Auto-scaled GUI**  
  Main window maximizes to the selected GUI screen’s available size (no fixed 1920×1080).

- **Display routing**  
  Dropdowns to choose which screen shows the Stimulus and which hosts the GUI; optional fullscreen for the stimulus.

- **Two independent delays**  
  - Lights delay → `lights_delay_s`  
  - Stimulus delay → `stim_delay_s`

- **Persistent stimulus window**  
  Created at app start and remains open; draggable on Windows when windowed.

- **Manual Trigger button**  
  Run a single trial instantly—no hardware required.
