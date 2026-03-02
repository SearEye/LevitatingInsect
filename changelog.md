# FlyPy / FlyAPI — Master Change Log  
Comprehensive Version History from Project Inception to v1.44.1  

This document records all major architectural, functional, and stability changes
to FlyAPI.py and the FlyPy experimental framework as discussed and iteratively
implemented across project chats and file revisions.

Ordering:  
Newest versions appear at the top.  
Oldest foundational versions appear at the bottom.  

All dates correspond to when the changes were discussed and implemented within
project chat logs and file snapshots.

---

## v1.44.1 — Thread-Safe Trial Execution & GIL Mitigation  
**Date: 2026-02-25**

**Primary Focus:** Stability under high-FPS load and GUI responsiveness.

**Changes:**
- All trials (manual trigger and hardware-triggered) execute in a dedicated
  background worker thread.
- Eliminated Qt GUI thread blocking during `Trigger Once`.
- Added cooperative `time.sleep(0)` yielding inside camera writer loops to
  mitigate GIL starvation.
- Resolved:
  - Stimulus “instant max size” artifact.
  - GUI freeze during manual trigger.
  - Stimulus frame-skipping under high camera throughput.
- Added explicit trial locking to prevent overlapping executions.

**Impact:**  
This version represents the first fully stable threading architecture suitable
for thesis-grade deployment.

---

## v1.44.0 — Deterministic Post-Render Stimulus Pipeline  
**Date: 2026-02-25**

**Primary Focus:** Elimination of concurrent video corruption.

**Changes:**
- Removed concurrent stimulus window capture during camera recording.
- Implemented deterministic offline stimulus rendering after camera clips complete.
- Stimulus video synthesized from timeline parameters rather than window capture.
- Preserved real-time presentation behavior.
- Eliminated AVI corruption caused by multi-writer contention.

**Impact:**  
Separated behavioral presentation from archival rendering.
Improved reproducibility and video integrity.

---

## v1.43.2 — Stimulus Image Scaling & Persistent Window Refinement  
**Date: 2026-02-23**

**Primary Focus:** Stimulus visual precision and usability.

**Changes:**
- Added aspect-ratio–preserving image scaling.
- Improved PNG stimulus handling (RGBA alpha support).
- Added persistent stimulus window option.
- Added screen selection and fullscreen toggle controls.
- Refined cubic easing for looming growth.

**Impact:**  
Improved perceptual smoothness and ensured cross-display compatibility.

---

## v1.36.0 — PySpin Device Serial Fix & Acquisition Stability  
**Date: 2026-02-23**

**Primary Focus:** Correct camera selection and acquisition control.

**Changes:**
- PySpin device selection now matches `DeviceSerialNumber`
  instead of `UniqueID`.
- Fixed repeated “Camera is already streaming” error spam.
- Corrected Advanced toggle signal signature.
- Ensured process-lifetime PySpin system handling.
- Standardized Desktop log directory with end timestamp.

**Impact:**  
Resolved camera misassignment across dual Blackfly setups.
Improved startup reliability.

---

## v1.35.4 — Acquisition Safety & GUI Enhancements  
**Date: 2026-01-28**

**Primary Focus:** Acquisition robustness.

**Changes:**
- Added guard against redundant `BeginAcquisition()` calls.
- Improved logging consistency.
- Stabilized preview toggling behavior.
- Standardized full-resolution recording (preview off by default).
- Added FPS probe utility.

**Impact:**  
Established stable dual-camera recording baseline.

---

## v1.30.x — Unified Trigger → Camera → Stimulus Integration  
**Date: 2026-01 (Early)**

**Primary Focus:** Core system unification.

**Changes:**
- Integrated hardware beam-break trigger (“T” token via serial).
- Linked trigger to dual camera acquisition.
- Added initial PsychoPy looming stimulus implementation.
- Implemented CSV trial logging.
- Established folder-based per-trial outputs.

**Impact:**  
First functional full trigger-to-record pipeline.

---

## v1.2x — Blackfly PySpin Backend Introduction  
**Date: 2025 (Late)**

**Primary Focus:** High-speed camera integration.

**Changes:**
- Added Spinnaker / PySpin backend.
- Configurable ROI.
- Exposure control.
- Hardware Line0 trigger mode.
- Stream buffer optimization (NewestOnly).

**Impact:**  
Transitioned from OpenCV-only prototype to high-speed research platform.

---

## v1.1x — OpenCV Baseline Prototype  
**Date: 2025 (Mid-Late)**

**Primary Focus:** Proof-of-concept behavioral recording.

**Changes:**
- Single and dual OpenCV camera recording.
- Basic GUI controls.
- Manual trigger button.
- Fixed-duration looming circle stimulus.
- Simple AVI output.

**Impact:**  
Established minimal viable behavioral assay platform.

---

## v1.0 — FlyPy Foundational Prototype  
**Date: 2025 (Early Development Phase)**

**Primary Focus:** Concept validation.

**Changes:**
- Manual recording initiation.
- Single-camera OpenCV capture.
- Basic growing circle stimulus.
- No hardware trigger integration.
- No CSV metadata logging.

**Impact:**  
Initial demonstration that a software-controlled looming stimulus
could be synchronized with behavioral video capture.

---

# Development Trajectory Summary

FlyPy evolved through five major architectural phases:

1. Prototype Phase — Manual OpenCV stimulus/record.
2. Hardware Integration Phase — Arduino trigger + logging.
3. High-Speed Transition Phase — PySpin / Blackfly integration.
4. Stability Phase — Device selection, acquisition safety.
5. Deterministic Phase — Thread-safe execution + post-render stimulus pipeline.

---

# Current Release Status

FlyAPI v1.44.1  
Designated: Thesis Stable Release Candidate  

No known race conditions.  
No GUI blocking behavior.  
No concurrent writer corruption.  
Time-accurate behavioral synchronization preserved.

---
