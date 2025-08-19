# -*- mode: python ; coding: utf-8 -*-

"""
PyInstaller spec for the FlyPy-Full (with PsychoPy) one-folder build.

Key points:
- Collects data/binaries/hiddenimports for psychopy, cv2, numpy, serial, pyglet, sounddevice, soundfile.
- Bundles the repo's temp/ folder (drivers, helper scripts) into dist/FlyPy/temp/.
- Includes README.md and LICENSE in the root of the portable app.
- Windowed EXE (no console). Change console=True if you want a console window.

If you rename FlyAPI.py, update the 'script' below accordingly.
"""

import os
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Packages we need to sweep thoroughly (data, binaries, hidden imports)
PKGS = ["psychopy", "cv2", "numpy", "serial", "pyglet", "sounddevice", "soundfile"]

datas = []
binaries = []
hiddenimports = ["serial.tools.list_ports"]  # explicit helper

for pkg in PKGS:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# Always include project docs
datas += [
    ("README.md", "."),
    ("LICENSE", "."),
]

# Bundle the entire temp/ folder (drivers + scripts) if present
temp_src = os.path.join(os.path.abspath("."), "temp")
if os.path.isdir(temp_src):
    # Put it under dist/FlyPy/temp/
    datas.append((temp_src, "temp"))

a = Analysis(
    ["FlyAPI.py"],              # <-- entry script
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="FlyPy",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                 # leave True unless UPX causes issues on some PCs
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,            # windowed app
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,                # add an .ico here if you like
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="FlyPy",             # -> dist/FlyPy/
)
