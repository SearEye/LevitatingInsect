"""
auto_update.py — tiny helper to check GitHub Releases, download newest portable build,
and stage a Windows-safe self-update (applies after app exit).

Usage from FlyAPI.py:
    import auto_update
    auto_update.check_and_prompt(repo="your-user/your-repo",
                                 current_version=__version__,
                                 asset_name="FlyPy-Full.zip",
                                 channel="releases")

Notes:
- Designed for PyInstaller "one-folder" builds (portable folder).
- On update found: downloads zip to %TEMP%, writes 'update_on_restart.bat'
  that replaces files *after* the app exits, then relaunches FlyPy.exe.
- Minimal deps: stdlib only (urllib, zipfile, shutil).
"""

import os
import sys
import json
import time
import shutil
import zipfile
import tempfile
import urllib.request
from urllib.error import URLError, HTTPError

API_LATEST  = "https://api.github.com/repos/{repo}/releases/latest"
API_RELEASE = "https://api.github.com/repos/{repo}/releases"  # all releases (if you want to filter channels)

UA = "FlyPy-Updater/1.0 (+https://github.com)"

def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode("utf-8")

def _parse_ver(s: str):
    """Very small semantic-ish version parser: 'v1.2.3' -> (1,2,3)."""
    s = s.strip()
    if s.startswith("v") or s.startswith("V"):
        s = s[1:]
    parts = []
    for tok in s.split("."):
        try:
            parts.append(int(tok))
        except Exception:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])

def _is_newer(remote_tag: str, current: str) -> bool:
    return _parse_ver(remote_tag) > _parse_ver(current)

def _find_asset(assets, name: str):
    for a in assets:
        if a.get("name", "").lower() == name.lower():
            return a
    return None

def _download(url: str, dest: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)

def _app_root() -> str:
    """Return the folder that holds FlyPy.exe (PyInstaller) or the script directory."""
    if getattr(sys, "frozen", False):
        # PyInstaller one-folder: sys.executable is FlyPy.exe in dist/FlyPy/
        return os.path.dirname(sys.executable)
    # dev: script file location
    return os.path.dirname(os.path.abspath(sys.argv[0]))

def _write_update_bat(zip_path: str, app_root: str, exe_name: str = "FlyPy.exe") -> str:
    """Write a small .bat that replaces app_root with the zip contents and relaunches."""
    bat_path = os.path.join(app_root, "update_on_restart.bat")
    # We extract to a temp dir next to app_root, then robocopy over.
    temp_unpack = os.path.join(app_root, "_upd_unpack")
    with open(bat_path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("@echo off\n")
        f.write("setlocal EnableExtensions EnableDelayedExpansion\n")
        f.write("echo Waiting for the app to exit...\n")
        f.write("ping -n 3 127.0.0.1 >nul\n")
        f.write("set ZIP=\"{}\"\n".format(zip_path))
        f.write("set APPROOT=\"{}\"\n".format(app_root))
        f.write("set UNPACK=\"{}\"\n".format(temp_unpack))
        f.write("rmdir /S /Q %UNPACK% 2>nul\n")
        f.write("mkdir %UNPACK%\n")
        f.write("echo Unpacking update...\n")
        f.write("powershell -NoProfile -Command \"Expand-Archive -Path %ZIP% -DestinationPath %UNPACK% -Force\" 2>nul\n")
        f.write("if not exist %UNPACK%\\* (\n")
        f.write("  echo PowerShell unzip failed, trying Python...\n")
        f.write(")\n")
        f.write("if not exist %UNPACK%\\* (\n")
        f.write("  \"{}\" - <<PY\n".format(sys.executable if not getattr(sys, "frozen", False) else "python"))
        f.write("import zipfile,sys,os; z=zipfile.ZipFile({}); d={}; z.extractall(d); z.close()\n".format("os.environ['ZIP'][1:-1]", "os.environ['UNPACK'][1:-1]"))
        f.write("PY\n")
        f.write(")\n")
        f.write("echo Applying update...\n")
        # Robocopy to copy all files (mir /xo) while keeping running bat
        f.write("robocopy %UNPACK% %APPROOT% /E /XO >nul\n")
        f.write("echo Cleaning up...\n")
        f.write("rmdir /S /Q %UNPACK% 2>nul\n")
        f.write("del %ZIP% 2>nul\n")
        f.write("echo Relaunching...\n")
        f.write("start \"\" \"%APPROOT%\\{}\"\n".format(exe_name))
        f.write("exit /b 0\n")
    return bat_path

def check_and_prompt(repo: str, current_version: str, asset_name: str, channel: str = "releases"):
    """
    Check GitHub for the latest release. If newer than `current_version`, download the named asset
    to a temp file and create an 'update_on_restart.bat' in the app folder.

    Returns:
        (found, message) where found indicates an update was staged or not.
    """
    try:
        meta = json.loads(_get(API_LATEST.format(repo=repo)))
    except (URLError, HTTPError) as e:
        return False, f"Could not contact GitHub: {e}"
    except Exception as e:
        return False, f"Update check failed: {e}"

    remote_tag = str(meta.get("tag_name", "")).strip()
    if not remote_tag:
        return False, "No releases found for this repository."

    if not _is_newer(remote_tag, current_version):
        return False, f"Up to date (current {current_version}, latest {remote_tag})."

    asset = _find_asset(meta.get("assets", []), asset_name)
    if not asset:
        return False, f"New release {remote_tag} found, but asset '{asset_name}' was not attached."

    url = asset.get("browser_download_url")
    if not url:
        return False, "Asset download URL missing."

    # Download to temp
    tmpdir = tempfile.mkdtemp(prefix="flypy_upd_")
    dest_zip = os.path.join(tmpdir, asset_name)
    try:
        _download(url, dest_zip)
    except Exception as e:
        return False, f"Failed to download update: {e}"

    # Write helper .bat into the app folder
    app_root = _app_root()
    bat = _write_update_bat(dest_zip, app_root)

    return True, f"Update {remote_tag} downloaded. Close the app to finish installing.\n\n" \
                 f"A helper script has been prepared:\n{bat}\n" \
                 f"It will apply the update and relaunch."
