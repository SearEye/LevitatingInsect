# FlyAPI Change Log

## v1.44.1
- All trials run in background worker thread.
- Cooperative yielding added to camera writer loops.
- Fixed GUI freeze during Trigger Once.
- Fixed stimulus instant-max-size artifact.
- Deterministic timing preserved under high FPS load.

## v1.44.0
- Removed concurrent stimulus window capture.
- Implemented post-record deterministic stimulus rendering.
- Eliminated AVI corruption under load.

## v1.43.x
- Image scaling and aspect-ratio controls.
- Persistent stimulus window logic.

## v1.36.0
- PySpin device selection via DeviceSerialNumber.
