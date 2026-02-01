# FlyPy (FlyAPI.py) — v1.36.1

Unified trigger → **cameras + lights + looming stimulus** for Drosophila escape-behavior assays.

- **GUI:** PyQt5 (one window), with *Simulation Mode*, *Probe Max FPS*, presets, camera previews.
- **Cameras:** FLIR Blackfly via **PySpin** (preferred) or **OpenCV** webcams as fallback.
- **Stimulus:** PsychoPy (preferred) or OpenCV fallback. Supports **expanding circle** or a **PNG** that scales like the circle.
- **Hardware:** Arduino/Elegoo UNO sends `'T'` on trip‑beam to trigger. Optional lights lane control via serial text commands.
- **Data:** Per‑trial videos + `trials_log.csv` metadata. Keep genotype tracking in Excel template (included).

> If you’re only testing the UI without hardware, enable **Simulation Mode** (periodic triggers).

---

## 1) Installation & Setup

### 1.1. Recommended environment
- **OS:** Windows 10/11 (for PySpin/Spinnaker). Linux works with the correct Spinnaker build; macOS not officially supported by PySpin.
- **Python:** 3.9–3.11
- **GPU:** not required; CPU is fine.
- **Monitors:** at least 2 if you want the GUI and stimulus on separate screens.

### 1.2. Python dependencies
Create/activate a virtual environment, then:

```bash
pip install -r requirements.txt
```
`requirements.txt` includes:
- `PyQt5>=5.15`
- `numpy>=1.23`
- `opencv-python>=4.5`
- `psychopy>=2023.2`
- `pyserial>=3.5`

#### PySpin / Spinnaker (for FLIR Blackfly)
1. Install the **Spinnaker SDK** for your OS from Teledyne FLIR.
2. During install, include the **Python bindings**.
3. Ensure the Python wheels provided by the SDK are installed for your Python version (you may run something like `pip install spinnaker_python‑<...>.whl` from the SDK installer folder).  
4. Make sure the SDK runtime libraries are in `PATH` (Windows) or your linker path (Linux).  
If PySpin is not available, the app will fall back to OpenCV cameras.

### 1.3. Run
```bash
python FlyAPI.py
```
On first launch you’ll see the main window. If you want a quick dry run, check **“Test/Simulation Mode”** and press **Start**.  

---

## 2) Hardware Overview

### 2.1. Arduino / Elegoo UNO
- The sketch should send a **single line** with the capital letter **`T`** followed by newline when the **laser tripwire** is broken (or when a digital button is pressed for testing).
- Connect UNO via USB. The app auto‑detects common CH340/UNO serial ports.

**Minimal trigger sketch (pseudo‑Arduino):**
```cpp
// Pseudocode: send 'T\n' when the beam is broken
const int SENSOR_PIN = 2;  // digital or interrupt-capable
const int ACTIVE_LOW = 1;  // set according to sensor board

void setup() {
  Serial.begin(115200);
  pinMode(SENSOR_PIN, INPUT_PULLUP);
}

void loop() {
  int s = digitalRead(SENSOR_PIN);
  bool broken = ACTIVE_LOW ? (s == LOW) : (s == HIGH);
  if (broken) {
    Serial.println("T");
    delay(200); // de-bounce
  }
}
```

### 2.2. Laser tripwire (breadboard)
- Use a **break‑beam IR pair** or a **laser diode + photodiode/LDR**.
- Condition the sensor output to clean 0/5 V transitions (comparator module recommended).  
- Wire sensor output → **UNO D2** (or any digital pin you prefer and match in code).  
- **GND** to UNO GND, **VCC** to 5V (or module spec).

### 2.3. Lane lights (optional)
- The app can emit the following serial *strings*: `LIGHT ON`, `LIGHT OFF`, `MARK START`, `MARK END`, `STIM`.  
- On the Arduino side, parse incoming serial and drive a MOSFET + LED bar/relay accordingly.
- Keep lights on a separate power rail; tie grounds.

### 2.4. Cameras
- **Blackfly (preferred):** Install Spinnaker, then connect cameras via USB3.0. Use **PySpin** in the GUI.
- **Fallback webcams:** Select **OpenCV** backend if PySpin is unavailable.

---

## 3) GUI Walk‑through

**Top controls**
- **Quick Preset:** Apply common Blackfly or OpenCV settings.
- **Probe Max FPS:** Measures achievable FPS for both cameras (3 s) and suggests a stable target.
- **Refresh Cameras:** Rebuilds the device list (PySpin serials + OpenCV indices).

**Core buttons**
- **Start**: begin trigger‑watch loop (from Arduino or Simulation Mode).  
- **Stop**: stop the loop.  
- **Trigger Once**: run a single trial manually.

**General panel**
- **Test/Simulation Mode:** emit periodic triggers with adjustable interval.  
- **Output folder:** where videos and logs are saved.  
- **Video format/codec:** e.g., AVI/MJPG (fast, huge) or MP4/mp4v.  
- **Recording duration (s).**

**Stimulus & Timing**
- **Stimulus total time (s)**, **start radius px**, **end radius px**.  
- **Background shade** (0=black, 1=white).  
- **Delays**: *record → lights ON*, *record → stimulus ON*.  
- **Stimulus Type**: **Circle** or **PNG (scaled)**.  
- **Stimulus PNG + Keep aspect.**  
- **Keep stimulus window open while running** (recommended for stability).

**Display & Windows**
- **Stimulus display screen:** human‑readable labels like `1: Dell 1920×1080 …`. Click **Refresh Displays** if you plug screens in/out.  
- **GUI display screen:** where the control window sits.  
- **Borderless fullscreen (F11‑style):** toggles true fullscreen for the stimulus.  
- **Pre‑warm stimulus window at launch:** creates the window before any trials.

**Camera 0/1 panels**
- **Backend:** OpenCV or PySpin.  
- **Device:** pick from discovered devices or **type manual index/serial**.  
- **Target FPS** and **Advanced…** (ROI width/height, exposure µs, hardware trigger).  
- **Show Preview:** live thumbnail; the FPS label shows driver‑reported frame cadence.

---

## 4) Scientific Procedures (Looming Response Assay)

### 4.1. Preparation
1. **Room & rig**: consistent lighting, no vibrations, fixed chamber geometry.  
2. **Monitors**: set stimulus display to the monitor facing the flies. Turn on **Borderless fullscreen**.  
3. **Calibration**:
   - Verify monitor **refresh rate** (≥60 Hz; 120+ is better).
   - Use a ruler/known reference to map **pixels → mm** on the stimulus monitor.
   - Optionally compute **visual angle** if distance from fly to screen is fixed:  
     \( \theta = 2\arctan\tfrac{size/2}{distance} \).  
4. **Cameras**:
   - Set ROI to avoid bandwidth saturation (e.g., 640×512).  
   - Target FPS ≤ 90% of probed max.  
   - Exposure ≲ 85% of frame period.  
5. **Laser tripwire**: verify serial `'T'` events by opening Arduino Serial Monitor (close it before running FlyPy!).  
6. **Lights lane**: verify **LIGHT ON/OFF** handling on Arduino side (optional).

### 4.2. Trial structure used by FlyPy
- On trigger:
  1. **MARK START** (serial)
  2. **Start recording** both cameras (to duration you set)
  3. **Lights delay** (if any) → `LIGHT ON`
  4. **Stimulus delay** (if any) → `STIM` + show looming
  5. Continue recording until `record_duration_s` elapses from `MARK START`
  6. `LIGHT OFF` + `MARK END`
- Output: `/FlyPy_Output/YYYYMMDD/trial_YYYY‑mm‑dd_HH‑MM‑SS/` with `cam0.*`, `cam1.*` and row in `trials_log.csv`.

### 4.3. Benchmarks: shapes × sizes × grow speeds
To compare behavioral response probability and latency across stimuli:

1. **Plan a grid** of conditions (example):
   - Shapes: **circle**, **PNG** (e.g., hawk silhouette, looming square).  
   - Start radii: 4, 8, 16 px.  
   - End radii: 200, 300, 400 px.  
   - Durations: 0.25, 0.5, 1.0, 1.5 s.  
2. **Randomize trial order** (avoid adaptation).  
3. **Replicates**: ≥10–20 per condition to estimate response probability.  
4. **Document** condition labels in a sheet (see *Excel schema* below).  
5. **Run** each trial (either via tripwire or **Trigger Once**).  
6. **Score** (manually or with tracking): takeoff occurrence (Y/N), latency (ms), wing elevation, jump distance.  
7. **Analyze**: plot probability vs. size/speed; compare shapes; compute median latency and IQR.

**Tip**: Use the **PNG** mode with *Keep aspect* ON to mimic circle scaling for arbitrary silhouettes.

### 4.4. Multi‑fly vs single‑fly
- For single‑fly assays, center the animal and ensure the looming origin is aligned with the fly’s field of view.
- For multi‑fly, consider adjusting ROI and camera angle; ensure individuals are trackable.

---

## 5) Data & Metadata

### 5.1. Folder layout
```
FlyPy_Output/
  20260131/
    trial_2026-01-31_22-12-43/
      cam0.avi
      cam1.avi
    trial_2026-01-31_22-14-08/
      cam0.avi
      cam1.avi
  trials_log.csv
```

### 5.2. `trials_log.csv` columns
The app writes a row per trial with (abbrev):
- `timestamp`, `trial_idx`
- `cam0_path`, `cam1_path`
- `record_duration_s`
- `lights_delay_s`, `stim_delay_s`, `stim_duration_s`
- `stim_screen_index`, `stim_fullscreen`
- `stim_kind`, `stim_png_path`, `stim_png_keep_aspect`, `stim_keep_window_open`
- `cam0_backend`, `cam0_ident`, `cam0_target_fps`, `cam0_w`, `cam0_h`, `cam0_exp_us`, `cam0_hwtrig`
- `cam1_backend`, `cam1_ident`, `cam1_target_fps`, `cam1_w`, `cam1_h`, `cam1_exp_us`, `cam1_hwtrig`
- `video_preset_id`, `fourcc`

### 5.3. Genotypes & cohorts (Excel)
Use **`Genotypes.xlsx`** (template included) and keep one row per *group or vial* you plan to test. Suggested columns:
- `StockID`, `LineName`, `Genotype` (full), `Sex`, `Age_Days`
- `Vial_ID`, `Date_Bred`, `Date_Anesthetized`, `Cohort_Notes`
- `Temp_C`, `FoodBatch`, `Handling_Notes`
- `Operator`, `Rig_ID`
- `Condition_Label` (e.g., “PNG_hawk_r0=8_r1=300_dur0.5”)
- `Trial_Indices` (e.g., “12;15;18” once trials are completed)

> Keep the condition labels consistent with how you randomize shape/size/speed.

---

## 6) Troubleshooting

**PySpin import error**
- Confirm **Spinnaker SDK** is installed and you installed the matching Python wheel for your Python version.
- Ensure runtime libraries are in `PATH` (Windows) or loader path (Linux).
- If still failing, switch backend to **OpenCV** in the GUI as a temporary workaround.

**OpenCV camera not found**
- Index might be wrong. Click **Refresh Cameras** and select discovered webcam(s).
- Another app may be using the camera. Close it.

**Serial/Arduino not detected**
- Close **Arduino Serial Monitor** and any terminal apps.
- Try a different USB port/cable. Verify that **CH340/UNO** driver is installed.
- Use **Simulation Mode** to validate software while debugging hardware.

**Stimulus on wrong screen / not full borderless**
- Go to **Display & Windows** → choose **Stimulus display screen**; click **Refresh Displays** after plugging monitors in.
- Enable **Borderless fullscreen (F11‑style)**.
- If PsychoPy window fails, the system will fall back to OpenCV; some window managers ignore borderless hints—try setting that monitor as **primary**.

**Low FPS / dropped frames**
- Reduce camera **ROI** size or lower target FPS.
- Use **Probe Max FPS** and set Target FPS ≤ 90% of measured max.
- Use **MJPG/AVI** for lighter CPU encode during acquisition.

**PNG won’t load / shows as circle**
- Verify PNG path. If load fails, FlyPy falls back to circle to avoid crashing.

**Nothing happens when I break the beam**
- Confirm your Arduino prints `T` + newline.
- In Simulation Mode, verify that triggers appear periodically; then switch back to serial mode once hardware is fixed.

---

## 7) Safety & Good Practices
- Use **IR break‑beam** or low‑power laser modules; avoid specular reflections into eyes.
- Avoid excessive lighting brightness; document illuminance if relevant.
- Ensure chamber ventilation and humane handling of flies.

---

## 8) Reproducibility Checklist
- Fixed monitor distance and pixel‑to‑mm mapping.
- Logged **genotype**, **age**, **sex**, **temperature**, **food batch**.
- Randomized stimulus order.
- Stable camera FPS (probe before sessions).
- Trial timing verified (lights/stim delays).
- Backups of `trials_log.csv` and `Genotypes.xlsx` after each day.

---

## 9) Quick Start (5 minutes)
1. Install deps (`pip install -r requirements.txt`).  
2. (Optional) Install Spinnaker & PySpin for Blackfly.  
3. Run `python FlyAPI.py`.  
4. In **General**: set output folder.  
5. In **Stimulus & Timing**: choose **Circle** or **PNG**, set duration and sizes.  
6. In **Display & Windows**: pick **Stimulus display screen**, enable **Borderless fullscreen**.  
7. In **Cameras**: select devices (PySpin serials or OpenCV indices), set target FPS.  
8. Enable **Simulation Mode** (first test). Click **Start**.  
9. Watch a few trials, then disable Simulation Mode and connect Arduino + break‑beam.  
10. Run real trials; fill **Genotypes.xlsx** as you go.

---

**Version:** v1.36.1  
**License:** See `LICENSE` (MIT/Apache-style as applicable).  
**Contacts:** Lab maintainer / repository owner.
