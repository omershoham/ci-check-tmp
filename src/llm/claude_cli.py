"""Claude Code CLI provider: headless `claude -p` on the user's subscription.

Runs on CLAUDE_CODE_OAUTH_TOKEN. ANTHROPIC_API_KEY is stripped from the subprocess
environment so billing never silently falls back to per-token API usage. We do NOT
pass `--bare` (it disables OAuth/keychain auth and forces API-key billing).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

from .. import config
from . import Response


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    if config.get("claude.force_subscription", True):
        env.pop("ANTHROPIC_API_KEY", None)
    token = config.oauth_token()
    if token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    return env


def _claude_exe() -> str:
    """Resolve the launcher to a full path (on Windows it is an npm .cmd shim)."""
    return shutil.which("claude") or "claude"


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the process AND its children. The npm shim's real `node` grandchild
    otherwise keeps running (burning subscription) and holds the stdout pipe open."""
    try:
        import psutil  # optional; cross-platform tree kill
        try:
            parent = psutil.Process(proc.pid)
            for ch in parent.children(recursive=True):
                ch.kill()
            parent.kill()
            return
        except psutil.Error:
            pass
    except ImportError:
        pass
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           capture_output=True, timeout=15)
        else:
            os.killpg(os.getpgid(proc.pid), 9)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


class ClaudeCLIProvider:
    name = "claude_cli"
    supports_sessions = True
    supports_web_search = True      # the CLI's WebSearch/WebFetch tools

    @property
    def model_label(self) -> str:
        # Unchanged from before providers existed, so every historical report's
        # `model` value stays comparable with new ones.
        return str(config.get("claude.model", "claude-opus-4-8"))

    def ready(self) -> tuple[bool, str]:
        exe = shutil.which("claude")
        if not exe:
            return False, "the `claude` CLI is not on PATH (install Claude Code, or pick another llm.provider)"
        if not config.oauth_token():
            return False, "CLAUDE_CODE_OAUTH_TOKEN is unset — run `claude setup-token` and put it in .env"
        return True, f"claude CLI at {exe} · subscription token set · {self.model_label}"

    def complete(self, prompt: str, *, tools: str | None = None, timeout: float | None = None,
                 extra_args: list[str] | None = None, model: str | None = None) -> Response:
        model = model or config.get("claude.model", "claude-opus-4-8")
        allowed = tools if tools is not None else config.get("claude.allowed_tools", "WebSearch,WebFetch")
        timeout = timeout or config.get("claude.timeout_seconds", 600)
        # Prompt via stdin, not argv: the .cmd shim mangles long multi-line args.
        cmd = [_claude_exe(), "-p", "--output-format", "json", "--allowedTools", allowed,
               "--model", model] + (extra_args or [])
        popen_kw = {}
        if os.name != "nt":
            popen_kw["start_new_session"] = True    # so killpg reaches the whole tree
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                env=_subprocess_env(), **popen_kw)
        try:
            out, err = proc.communicate(input=prompt, timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            try:
                proc.communicate(timeout=10)
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(f"claude timed out after {timeout}s")
        if proc.returncode != 0:
            raise RuntimeError(f"claude exited {proc.returncode}: {(err or '')[:500]}")
        raw = (out or "").strip()
        text, sid = raw, None
        try:
            env = json.loads(raw)
            if isinstance(env, dict):
                text = env.get("result", raw)
                sid = env.get("session_id")
        except json.JSONDecodeError:
            pass
        return Response(text=text, raw=raw, session_id=sid, model=model)
