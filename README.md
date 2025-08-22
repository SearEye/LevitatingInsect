FlyPy — Trigger → Cameras + Lights + Looming Stimulus (All-in-One GUI)

FlyPy coordinates a trigger (hardware, simulated, or manual) to:

Record two cameras in sync to disk

Turn lights on/off (Elegoo/Arduino over serial, or simulated)

Present a looming stimulus (growing black dot) on a selected monitor

Log a CSV row per trial with file paths, timing, FPS, and settings

The app ships with a single window GUI (plain-English controls + tooltips), live previews for both cameras, codec presets, and optional auto-update from GitHub.

Version hint: the current app version is defined in FlyAPI.py as __version__ = "1.3.0".

What’s New (since 1.3.x)

Auto-update from GitHub: Help → Check for Updates… downloads the latest portable build (from your repo’s Releases) and stages it for installation after exit.

Auto-scaled GUI: the main window now maximizes to the selected GUI screen’s available size (no fixed 1920×1080).

Display routing: dropdowns to choose which screen shows the Stimulus and which screen hosts the GUI; optional fullscreen for the stimulus.

Two independent delays (relative to recording start):

Lights delay → lights_delay_s

Stimulus delay → stim_delay_s

Persistent stimulus window: created at app start and stays open until you quit; draggable on Windows when windowed (not fullscreen).

Manual Trigger button: run a single trial instantly—no hardware required.

Install & Launch
Quick start (Python installed)
python FlyAPI.py


On launch, choose Simulation Mode:

Yes → triggers from a timer (sim_trigger_interval)

No → triggers from hardware (Elegoo/Arduino sends T over serial)

Windows portable build (no Python required)

Use the provided PyInstaller setup:

build_flypy_full.bat — builds a one-folder “FlyPy-Full” (bundles Python + PsychoPy)

FlyPy_full.spec — reproducible PyInstaller spec (collects Qt, OpenCV, PsychoPy assets)

Run:

build_flypy_full.bat


Find the portable app at:

dist\FlyPy\FlyPy.exe


The build script also puts the official CH340/CH341 USB-serial driver in:

dist\FlyPy\temp\drivers\CH341SER.EXE


If Windows doesn’t recognize your Elegoo board, run:

dist\FlyPy\temp\scripts\install_drivers.bat

Hardware Overview (Elegoo/Arduino)

Trigger input: your sensor (laser break, button, etc.) toggles a digital pin; the microcontroller emits T via serial when triggered.

Serial protocol (already supported):

Host → Board: START, STIM, END, PULSE <ms>, LIGHT ON, LIGHT OFF

Board → Host: T (one line per trigger)

The included ElegooFlyPySync.ino sketch is compatible with the above—no changes required.

Auto-Update from GitHub (portable builds)

The app can check your GitHub Releases and stage a self-update.

Menu: Help → Check for Updates…

If a newer release is found with asset (e.g., FlyPy-Full.zip), FlyPy downloads it to a temp folder and prepares update_on_restart.bat inside the app directory.

Close the app (or click “Close & Update” if you wire a button) to apply the update and relaunch.

Developer setup (one-time):

Publish portable builds as Release assets (e.g., FlyPy-Full.zip).

In FlyAPI.py → MainApp.on_check_updates, set:

REPO = "your-github-user/your-repo"
ASSET_NAME = "FlyPy-Full.zip"


Keep __version__ in FlyAPI.py up-to-date for comparisons.

The updater is implemented in auto_update.py (stdlib only; no extra installs).

GUI Cheat Sheet
Top bar

Start — begin watching for triggers (hardware or simulated)

Stop — stop watching

Trigger Once (Manual) — run one trial immediately

Apply Settings — push current GUI values into the running app

Help → Check for Updates… — query GitHub Releases and stage an update

General

Interval between simulated triggers (seconds) → sim_trigger_interval

Output folder for all trials → output_root

Video file format / codec → video_preset_id (updates fourcc)

Recording duration per trigger (seconds) → record_duration_s

Stimulus & Timing

Stimulus display duration (seconds) → stim_duration_s

Starting / Final dot radius (px) → stim_r0_px, stim_r1_px

Background shade (0=black, 1=white) → stim_bg_grey

Delay from recording start → lights ON (seconds) → lights_delay_s (NEW)

Delay from recording start → stimulus ON (seconds) → stim_delay_s (NEW)

Timing model (per trial)

Cameras start recording (time 0)

After lights_delay_s → lights ON

After stim_delay_s → looming stimulus ON (runs for stim_duration_s)

Cameras finish; lights OFF; row logged in CSV

If both delays are set, they’re applied from the same origin (recording start). The app staggers waits so each event occurs at its requested absolute time.

Display & Windows

Stimulus display screen → stim_screen_index (NEW)

GUI display screen → gui_screen_index (NEW)

Stimulus fullscreen on selected screen → stim_fullscreen (NEW)

Window persistence

The stimulus window is created as soon as the app launches and stays open.

If windowed (fullscreen unchecked), it’s a standard OS window—drag-and-drop between monitors works on Windows.

If fullscreen, it pins to the selected monitor.

Cameras (per camera)

OpenCV device index → cam0_index / cam1_index

Target recording FPS → camN_target_fps

Labels show driver-reported FPS, measured preview FPS, and target FPS (diagnostics).

Outputs & Logging

Videos:

<output_root>/<YYYYMMDD>/<timestamp>_trial####_cam0.<ext>
<output_root>/<YYYYMMDD>/<timestamp>_trial####_cam1.<ext>


CSV log (<output_root>/trials_log.csv) columns:

trial, timestamp,
cam0_path, cam1_path,
record_duration_s,
lights_delay_s, stim_delay_s, stim_duration_s,
stim_screen_index, stim_fullscreen,
cam0_target_fps, cam1_target_fps,
video_preset_id, fourcc


If a requested FOURCC/container fails on the local system, FlyPy automatically falls back to .mp4 with mp4v.

Tips & Troubleshooting

No hardware on hand? Use Simulation Mode = Yes or click Trigger Once (Manual).

No PsychoPy? The stimulus uses an OpenCV window (already implemented).

Wrong monitor? Use Display & Windows. In windowed mode, drag the stimulus where you want—it persists.

Codec errors or odd framerates? Prefer MP4 (mp4v) for broad compatibility.

Serial not found? Install the CH340/CH341 driver (see temp/ folder in the portable app).

File Map

FlyAPI.py — main app (GUI, cameras, stimulus, hardware, logging, updater hook)

auto_update.py — GitHub Releases checker & self-update stager (Windows-safe)

ElegooFlyPySync.ino — Arduino sketch (triggers + light/sync protocol)

FlyPy_full.spec — PyInstaller spec (FlyPy-Full, includes PsychoPy)

build_flypy_full.bat — one-folder build script (creates dist/FlyPy/)

requirements.txt — core dependencies (numpy, opencv-python, PyQt5, pyserial; psychopy optional at runtime)

LICENSE, README.md — docs & license

License

See LICENSE.
