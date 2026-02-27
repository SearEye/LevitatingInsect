# FlyPy (FlyAPI v1.44.1)

FlyPy is an integrated trigger-to-recording platform for high-speed behavioral capture of
Drosophila melanogaster escape responses under deterministic looming stimulus presentation.

Version: v1.44.1  
Primary file: FlyAPI.py  
Preferred backend: PySpin (FLIR Blackfly)  
Fallback backend: OpenCV  

---

## 1. System Overview

FlyPy unifies:

1. Hardware trigger acquisition (beam-break / Arduino serial token “T”)
2. Dual synchronized high-speed camera recording
3. Deterministic looming stimulus presentation
4. Time-accurate stimulus video rendering (post-camera write)
5. Trial-wise metadata logging (CSV)
6. GUI-controlled experimental configuration

All timing is defined in real seconds (time.perf_counter) to ensure reproducibility and
cross-system determinism.

---

## 2. Architectural Changes in v1.44.x

### v1.44.0
- Stimulus video is no longer captured concurrently with camera recording.
- Instead, stimulus video is deterministically rendered AFTER camera clips complete.
- Eliminates AVI corruption under high camera throughput.
- Ensures time-accurate stimulus reconstruction independent of GIL load.

### v1.44.1
- All trials (manual + hardware) execute in background worker threads.
- Qt GUI thread is never blocked.
- Camera writer loops include cooperative yielding to prevent GIL starvation.
- Fixes:
  - “Instant max size” stimulus artifact
  - Stimulus freeze on Trigger Once
  - GUI repaint stalls during high-FPS capture

This version is the current thesis-grade stable release.

---

## 3. Trigger Architecture

### Preferred Mode: OUTPUT TRIGGER (default)

Arduino (Elegoo UNO R3):
- Mode: ANALOG
- DO: OFF
- Serial token: "T"

FlyAPI:
- Listens for exact token match.
- Enforces minimum trigger interval (default 0.30 s).
- Debounced internally.

Simulation mode:
- Generates synthetic trigger events at defined intervals.
- Used for bench testing and debugging without hardware.

---

## 4. Camera Architecture

Preferred: FLIR Blackfly via PySpin (Spinnaker SDK)

Configuration:
- PixelFormat: Mono8 (fallback BayerRG8)
- ROI centered
- ExposureAuto: Off
- GainAuto: Off
- AcquisitionMode: Continuous
- TriggerMode:
    - Hardware: Line0
    - Manual trigger: Software trigger fallback
- StreamBufferHandlingMode: NewestOnly

OpenCV fallback:
- Used when PySpin unavailable
- MJPG fourcc default
- Intended for development, not high-precision experiments

---

## 5. Stimulus Architecture

Stimulus Types:
- Circle (default)
- Image (PNG/JPG/BMP/WebP)

Presentation:
- PsychoPy preferred
- OpenCV fallback

Timing:
- Stimulus onset = lights_delay + stim_delay
- Total trial time = record_duration
- Stimulus duration independent of recording duration

Post-Recording Rendering (v1.44.0+):
- Deterministic offline frame synthesis
- Frame rate default: 60 FPS
- Ensures time-accurate stimulus archive
- Prevents multi-writer contention

Stimulus easing:
- Cubic easing (k^3)
- Prevents perceptual stepping artifacts

---

## 6. Trial Flow

1. MARK START (Arduino)
2. Camera threads spawn
3. Stimulus presentation thread begins
4. Lights ON after configured delay
5. Stimulus ON after configured delay
6. Lights OFF
7. MARK END
8. Camera threads join
9. Stimulus video rendered offline
10. CSV metadata written

All durations are enforced using real elapsed wall time.

---

## 7. Output Structure

FlyPy_Output/
    YYYYMMDD/
        trial_TIMESTAMP/
            cam0.avi / .mp4
            cam1.avi / .mp4
            stimulus.avi / .mp4
    trials_log.csv

Metadata includes:
- Timing parameters
- Camera configuration
- Stimulus type and image path
- Codec
- Trigger mode
- ROI and exposure settings

---

## 8. Threading Model (v1.44.1)

Qt GUI Thread:
- UI rendering
- Preview
- Event loop

Trigger Loop Thread:
- Watches serial or simulation input

Trial Worker Thread:
- Executes run_one()

Camera Threads:
- One thread per camera for real-time writing

Stimulus Presentation Thread:
- Separate timing-driven presentation

Offline Stimulus Render:
- Executed after camera join

This prevents Qt blocking and timing drift.

---

## 9. Stability Guidelines

Recommended for Blackfly 522 FPS:
- ROI: 640×512
- Exposure: ≤ 1500 µs
- MJPG codec
- Record duration ≤ 3 s

Target FPS:
- Set to ~90% of probe_max_fps()

---

## 10. Thesis Release Status

FlyAPI v1.44.1 is designated:

"Thesis Stable Release Candidate"

It resolves:
- Concurrent AVI corruption
- GIL starvation artifacts
- GUI freeze on manual trigger
- Instant stimulus max-size jump

No known architectural race conditions remain.

---
