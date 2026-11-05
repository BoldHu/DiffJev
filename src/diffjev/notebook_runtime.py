"""Standard-library launcher shared by Colab notebooks and local notebooks.

The notebook kernel handles Drive/UI; research code runs in the pinned Python
environment. No installation, downloads, or model imports happen on import.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys


class NotebookRuntime:
    def __init__(self, root, python=None):
        self.root = Path(root).resolve()
        settings = self.root / "outputs/setup/runtime.json"
        saved = json.loads(settings.read_text()) if settings.exists() else {}
        self.python = str(python or os.environ.get("DIFFJEV_PYTHON") or saved.get("python") or sys.executable)
        self.logs = self.root / "outputs/setup/logs"

    def run(self, args, label="setup"):
        self.logs.mkdir(parents=True, exist_ok=True)
        if not label.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Log label must contain only letters, digits, hyphens or underscores")
        env = os.environ.copy()
        env["DIFFJEV_ROOT"] = str(self.root)
        env["PYTHONPATH"] = str(self.root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        with (self.logs / (label + ".log")).open("a", encoding="utf-8") as log:
            log.write("\nStarted " + datetime.datetime.now(datetime.timezone.utc).isoformat() + "\n")
            log.flush()
            process = subprocess.Popen([str(x) for x in args], cwd=self.root, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                status = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
            finally:
                process.stdout.close()
            log.write("Exit status: " + str(status) + "\n")
        if status:
            raise RuntimeError(f"{label} failed (exit {status}); see {self.logs / (label + '.log')}")

    def python_code(self, source, label="notebook"):
        if not Path(self.python).is_file():
            raise FileNotFoundError("Experiment Python is missing; rerun notebook 00 installation after a runtime reset")
        self.run([self.python, "-u", "-c", source], label)

    def save_settings(self):
        target = self.root / "outputs/setup/runtime.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        commit = None
        dirty = None
        try:
            revision = subprocess.run(["git","rev-parse","HEAD"],cwd=self.root,capture_output=True,text=True)
            status = subprocess.run(["git","status","--porcelain"],cwd=self.root,capture_output=True,text=True)
            if revision.returncode == 0:
                commit = revision.stdout.strip()
            if status.returncode == 0:
                dirty = bool(status.stdout.strip())
        except FileNotFoundError:
            pass
        target.write_text(json.dumps({"python": self.python,
            "code_commit":commit,"code_dirty":dirty,
            "requirements_sha256": hashlib.sha256((self.root/"requirements-lock.txt").read_bytes()).hexdigest()}, indent=2)+"\n")


def install_environment(root, env_dir, bootstrap_python=None):
    """Explicit opt-in installer. Uses a separate Python 3.12 worker environment."""
    bootstrap_python = bootstrap_python or sys.executable
    env_dir = Path(env_dir).resolve()
    target = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    launcher = NotebookRuntime(root, target)
    launcher.run([bootstrap_python,"-m","pip","install","uv==0.9.5"], "install_uv")
    if not target.exists():
        launcher.run([bootstrap_python,"-m","uv","venv","--python","3.12",str(env_dir)], "create_environment")
    launcher.run([bootstrap_python,"-m","uv","pip","install","--python",str(target),
                  "-r",str(launcher.root/"requirements-lock.txt")], "install_requirements")
    launcher.run([bootstrap_python,"-m","uv","pip","install","--python",str(target),
                  "--no-deps","-e",str(launcher.root)], "install_project")
    launcher.save_settings()
    return launcher
