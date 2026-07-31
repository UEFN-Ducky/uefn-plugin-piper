"""Pipe (Piper) — UEFN Ducky desktop plugin.

Runs neural TTS entirely on this machine with no API key: the Piper engine and each
voice model download once (from the public rhasspy/piper release + Hugging Face voice
repo), then everything works fully offline. Users can also drop their own Piper
``.onnx`` + ``.onnx.json`` models into the voices folder to add custom voices.

Wired into the app's provider-agnostic voice system:
  register(api) -> api.register_tts(synthesize)      # text -> WAV bytes
                -> api.register_tts_voices(list)      # downloaded + custom voices
"""

from __future__ import annotations

import base64
import io
import logging
import os
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

log = logging.getLogger("uefn.plugin.piper")

PLUGIN_ID = "piper"
DEFAULT_VOICE = "en_US-lessac-medium"

# Archived but permanently downloadable Windows build of the Piper engine.
PIPER_WIN_URL = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip"
# Voice models live in the public rhasspy/piper-voices repo; the path is derivable
# from a voice id like "en_US-amy-medium" -> en/en_US/amy/medium/<id>.onnx[.json].
VOICES_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

_HTTP_TIMEOUT = 300  # seconds — voice models can be tens of MB
_INSTALL_LOCK = threading.Lock()


def _appdata() -> Path:
    from backend.skills.store import appdata_dir

    return appdata_dir()


def piper_root() -> Path:
    return _appdata() / "piper_tts"


def bin_dir() -> Path:
    return piper_root() / "bin"


def voices_dir() -> Path:
    d = piper_root() / "voices"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _auto_download() -> bool:
    try:
        from frontend.ui_web.plugin_host_api import prefs_plugin_get

        prefs = prefs_plugin_get(PLUGIN_ID) or {}
        val = prefs.get("auto_download")
        return True if val is None else bool(val)
    except Exception:  # noqa: BLE001 — prefs are best-effort
        return True


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "UEFN-Ducky-Piper/1.0"})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
        return resp.read()


def _find_piper_exe() -> str | None:
    root = bin_dir()
    if not root.is_dir():
        return None
    name = "piper.exe" if sys.platform == "win32" else "piper"
    for path in root.rglob(name):
        if path.is_file():
            return str(path)
    return None


def _ensure_piper() -> str | None:
    """Return a path to the piper executable, downloading + extracting it once."""
    exe = _find_piper_exe()
    if exe:
        return exe
    if sys.platform != "win32":
        # Only the Windows engine is auto-installed for now; other OSes must drop a
        # piper binary into bin/ themselves.
        return None
    with _INSTALL_LOCK:
        exe = _find_piper_exe()
        if exe:
            return exe
        try:
            data = _download(PIPER_WIN_URL)
        except (urllib.error.URLError, OSError) as exc:
            log.warning("piper engine download failed: %s", exc)
            return None
        target = bin_dir()
        target.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    parts = Path(info.filename).parts
                    if ".." in parts or Path(info.filename).is_absolute():
                        continue  # zip-slip guard
                    dest = target.joinpath(*parts)
                    try:
                        if not dest.resolve().is_relative_to(target.resolve()):
                            continue
                    except (OSError, ValueError):
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(zf.read(info.filename))
        except (zipfile.BadZipFile, OSError) as exc:
            log.warning("piper engine extract failed: %s", exc)
            return None
        return _find_piper_exe()


def _voice_urls(voice_id: str) -> tuple[str, str] | None:
    """Derive the .onnx / .onnx.json download URLs from a standard piper voice id."""
    parts = voice_id.split("-")
    if len(parts) < 3:
        return None
    lang_region = parts[0]  # en_US
    name = parts[1]  # amy (may contain underscores: hfc_female)
    quality = parts[2]  # medium / high / x_low
    lang0 = lang_region.split("_")[0]  # en
    base = f"{VOICES_BASE}/{lang0}/{lang_region}/{name}/{quality}/{voice_id}"
    return base + ".onnx", base + ".onnx.json"


def _ensure_voice(voice_id: str) -> str | None:
    """Return a path to the voice .onnx, downloading it (and its .json) if missing."""
    onnx = voices_dir() / f"{voice_id}.onnx"
    if onnx.is_file():
        return str(onnx)
    if not _auto_download():
        return None
    urls = _voice_urls(voice_id)
    if not urls:
        return None  # custom voice must already be on disk
    url_onnx, url_json = urls
    with _INSTALL_LOCK:
        if onnx.is_file():
            return str(onnx)
        try:
            onnx_bytes = _download(url_onnx)
            json_bytes = _download(url_json)
        except (urllib.error.URLError, OSError) as exc:
            log.warning("piper voice %s download failed: %s", voice_id, exc)
            return None
        try:
            tmp = onnx.with_suffix(".onnx.part")
            tmp.write_bytes(onnx_bytes)
            (voices_dir() / f"{voice_id}.onnx.json").write_bytes(json_bytes)
            tmp.replace(onnx)
        except OSError as exc:
            log.warning("piper voice %s write failed: %s", voice_id, exc)
            return None
    return str(onnx)


def _synthesize(text: str, voice_id: str) -> dict[str, Any]:
    """text + voice id -> {ok, audio_base64, mime} (host plays the WAV)."""
    clean = " ".join((text or "").split())
    if not clean:
        return {"ok": False, "error": "Nothing to speak"}
    exe = _ensure_piper()
    if not exe:
        if sys.platform != "win32":
            return {"ok": False, "error": "Auto-install is Windows-only; drop a piper binary into the bin folder."}
        return {"ok": False, "error": "Could not install the Piper engine (needs internet once)."}
    voice = (voice_id or "").strip() or DEFAULT_VOICE
    model = _ensure_voice(voice)
    if not model:
        if not _auto_download():
            return {"ok": False, "error": f"Voice {voice!r} isn't downloaded and auto-download is off."}
        return {"ok": False, "error": f"Could not install voice {voice!r} (needs internet once)."}
    tmp_wav = ""
    try:
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="piper_")
        os.close(fd)
        creationflags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
        proc = subprocess.run(
            [exe, "-m", model, "-f", tmp_wav],
            input=clean.encode("utf-8"),
            cwd=str(Path(exe).parent),
            capture_output=True,
            timeout=120,
            creationflags=creationflags,
        )
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", "replace").strip()
            return {"ok": False, "error": f"Piper failed: {err[:200] or 'unknown error'}"}
        audio = Path(tmp_wav).read_bytes()
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Piper timed out generating audio"}
    except Exception as exc:  # noqa: BLE001 — never raise across the worker bridge
        return {"ok": False, "error": str(exc) or "Piper run failed"}
    finally:
        if tmp_wav:
            try:
                os.unlink(tmp_wav)
            except OSError:
                pass
    if not audio:
        return {"ok": False, "error": "Piper produced no audio"}
    return {
        "ok": True,
        "audio_base64": base64.b64encode(audio).decode("ascii"),
        "mime": "audio/wav",
    }


def _pretty_label(voice_id: str) -> str:
    parts = voice_id.split("-")
    if len(parts) >= 3:
        name = parts[1].replace("_", " ").title()
        return f"{name} — {parts[0]}, {parts[2]}"
    return voice_id


def _list_voices() -> list[dict[str, str]]:
    """Every voice present on disk: downloaded curated ones + user-dropped models."""
    out: list[dict[str, str]] = []
    try:
        for onnx in sorted(voices_dir().glob("*.onnx")):
            vid = onnx.stem
            out.append({"id": vid, "label": _pretty_label(vid)})
    except OSError as exc:
        log.info("piper list voices failed: %s", exc)
    return out


def _prefetch() -> None:
    """Warm the engine + default voice so the first spoken reply isn't a long wait."""
    try:
        if not _auto_download():
            return
        _ensure_piper()
        _ensure_voice(DEFAULT_VOICE)
    except Exception as exc:  # noqa: BLE001
        log.info("piper prefetch skipped: %s", exc)


def register(api) -> None:
    """Wire the Piper synthesizer + local voice lister into the host."""
    api.register_tts(_synthesize)
    register_voices = getattr(api, "register_tts_voices", None)
    if callable(register_voices):
        register_voices(_list_voices)
    threading.Thread(target=_prefetch, name="piper-prefetch", daemon=True).start()
    api.log("Pipe (Piper) voices registered")
