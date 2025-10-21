FlyPy – Levitating Insect Camera + Stimulus Rig

Unified GUI to trigger and record from two high-speed cameras (FLIR/Point Grey via PySpin or generic USB via OpenCV), drive lights, and present a looming “falling object” stimulus. Designed for fruit fly (Drosophila) escape experiments.

Repo: https://github.com/SearEye/LevitatingInsect

Main app: FlyAPI.py (PyQt5 GUI)
Latest tested version: v1.35.4 (Windows 10/11, Python 3.10)

Highlights (v1.35.4)

✅ Dual camera capture with per-camera backend selection: PySpin (Spinnaker) or OpenCV

✅ Device picker shows PySpin cameras by DeviceSerialNumber (e.g., 24102007, 24102017) – opens the correct camera

✅ Previews are optional (off by default). Full-resolution, high-FPS recording regardless of preview state

✅ Advanced camera settings (toggle panel): ROI (W/H), Exposure (µs), Hardware trigger (Line0)

✅ Looming stimulus (black dot on white) with start/end size, duration, screen selector, fullscreen toggle

✅ Simulation mode (timer-based triggers) for bench testing

✅ Max FPS probe utility

✅ Robust logging to Desktop\LevitatingInsect-main\logs with start/stop timestamps

✅ Handles common PySpin quirks:

“Camera is already streaming” is auto-suppressed

Keeps a process-lifetime Spinnaker System instance to avoid ReleaseInstance crashes

Contents

FlyAPI.py – the GUI application

requirements.txt – Python dependencies

requirements.lock.txt – pinned versions (optional)

auto_update.py – optional helper

ElegooFlyPySync.ino – microcontroller firmware (trigger/light sync)

FlyPy_RunOnly.bat – run launcher (Windows)

FlyPy_SetupOnly.bat / FlyPy_AllInOne.bat – convenience installers (Windows)

LICENSE, .gitignore, .gitattributes, README.md

Quick start (Windows)
1) Install prerequisites

Python 3.10 (64-bit) recommended

FLIR Spinnaker SDK + PySpin (for Blackfly etc.)

Install Spinnaker SDK (64-bit)

Ensure these folders are on your PATH (example):

C:\Program Files\FLIR Systems\Spinnaker\bin64\

C:\Program Files\FLIR Systems\Spinnaker\bin64\vs2015\ (or your compiler subfolder)

PySpin is provided by the SDK installer (wheel). Verify with python -c "import PySpin; print('ok')"

OpenCV & PyQt5 and friends:

From the repo root:

py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt


Tip: If you only use USB webcams, you can skip Spinnaker/PySpin.

2) Run

Double-click FlyPy_RunOnly.bat or:

.venv\Scripts\python.exe FlyAPI.py


Optional flags:

--simulate : run without hardware, timer-based triggers

--prewarm-stim : open the stimulus window at launch

Using the GUI
Top bar

Quick Preset → fast setup for Blackfly @ 522 fps / 300 fps or OpenCV baseline

Apply Preset, Probe Max FPS, Refresh Cameras

Controls

Start / Stop a trigger-watch loop

Trigger Once for a single trial

Apply Settings to push current UI state to the runtime

General

Test/Simulation Mode (timer triggers) and interval (s)

Output folder

Video format/codec (AVI/MJPG, AVI/XVID, MP4/mp4v)

Recording duration (s)

Stimulus & Timing (Falling Object / Growing Dot)

Stimulus total time (s) (growth duration)

Stimulus Start Size (radius px)

Stimulus End Size (radius px)

Background shade (0 black → 1 white; dot is black)

Delays: record→lights ON, record→stimulus ON

Display & Windows

Stimulus display screen (drop-down of monitors)

GUI display screen

Stimulus fullscreen (toggle)

Pre-warm stimulus window (optional)

Camera 0 / Camera 1 panels (independent)

Backend: OpenCV or PySpin

Device: list of detected devices

PySpin devices appear as: PySpin <DeviceSerialNumber> — <Model>

OpenCV devices as: OpenCV index N

Manual index/serial: override field

Target FPS

Advanced… (toggle)

ROI Width/Height (0 = max)

Exposure (µs)

Hardware trigger (Line0)

Show Preview (off by default)

Driver-reported FPS (approximate) label

Important: The two camera panels must reference different devices (e.g., PySpin serial 24102017 vs 24102007, or different OpenCV indices). Use Refresh Cameras after plugging in devices.

Recording & Output

Each trigger (manual or from the hardware/CH340/UNO) creates:

FlyPy_Output/<YYYYMMDD>/trial_<timestamp>/
  cam0.<ext>
  cam1.<ext>


Container/codec follows your “Video format / codec” choice. MJPG in AVI is the most robust for very high FPS.

A CSV trials_log.csv is kept at the root of FlyPy_Output with the per-trial config and file paths.

Logging

Session logs go to:

C:\Users\Murpheylab\Desktop\LevitatingInsect-main\logs\
    FlyPy_run_<START>.log.tmp
    FlyPy_run_<START>__ENDED_<END>.log


On clean or crash exit, the .tmp file is renamed with an __ENDED_<timestamp> suffix.

Triggers & Lights

The app can talk to an Elegoo/UNO (CH340) over serial for MARK START/END and LIGHT ON/OFF.

If no CH340/UNO is found, it simulates those messages and you can still run full trials.

Recommended FPS for Drosophila escape (wings)

To capture wing kinematics during escape, we recommend ≥ 750 fps (ideally 750–1000 fps).

If only onset/timing is required (not detailed wing motion), you can go lower; for full wingbeat envelopes, go higher.

Troubleshooting
Both previews show the same camera

Ensure distinct devices are selected:

PySpin: set different serials (e.g., 24102017 and 24102007)

OpenCV: set different indices (e.g., 0 and 1)

Click Refresh Cameras, then re-assign.

If you only have one physical device connected, the other panel may show “synthetic”.

PySpin warning on exit:

Spinnaker: Can't clear an interface because something still holds a reference [-1004]

The app keeps a process-lifetime Spinnaker System to avoid mid-run releases.

If you still see this at exit, it’s benign. If cameras don’t re-enumerate across runs, unplug/replug or power-cycle.

BeginAcquisition: Camera is already streaming

Now handled once internally. If image retrieval still fails, call Refresh Cameras or power-cycle that camera.

OpenCV warnings: DSHOW/MSMF “can’t be used to capture by index”

Try a different index or use the PySpin backend for FLIR cameras.

Video writer warnings: “write frame skipped / expected 1 channel but got 3”

The app writes color frames; this warning appears with odd system ffmpeg builds but files still save.
Switch to AVI/MJPG if you hit reliability issues.

Stimulus window errors on move/resize

If the second monitor is added/removed while running, re-select the target screen and Apply Settings.

Performance tips

Disk throughput matters at >500 fps. Use SSD/NVMe and MJPG AVI if stability is a concern.

Keep previews off during experiments to save CPU/GPU; recording is full-res regardless.

Use the Max FPS probe (3 s) to set Target FPS to ~90% of measured for stability.

Building / Development
git clone https://github.com/SearEye/LevitatingInsect
cd LevitatingInsect
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python FlyAPI.py --simulate


Command-line flags

--simulate – timer triggers without hardware

--prewarm-stim – open stimulus window on startup

Changelog (abridged)

v1.35.4

PySpin selection by DeviceSerialNumber → consistent multi-camera assignment (24102007 vs 24102017)

Suppress repeated “already streaming” BeginAcquisition warnings

Advanced… toggle fixed (Qt clicked(bool) signature)

Persistent Spinnaker System; safer shutdown; enriched logs

Preview toggles default OFF; full-res recording; FPS probe

Stimulus: black dot on white; start/end size; screen picker; fullscreen; simulation mode

(See logs in logs/ for exact run histories.)

License

See LICENSE in this repository.

Acknowledgements

FLIR/Point Grey Spinnaker/PySpin

OpenCV, PyQt5

PsychoPy (used when available; falls back to OpenCV otherwise)

Contact

Please open an issue on the GitHub repo with:

logs/FlyPy_run_...__ENDED_...log

Your Python & OS version

Camera models/serials and chosen backends

A short description of what you clicked before the problem occurred

Happy recording! 🪰📹
