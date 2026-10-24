"""
One-click launcher for MPG Studio (Windows, macOS, Linux).
IDLE: open this file and press F5.   Terminal: python run.py
Keep run.py, app.py and auto-mpg.csv (optional) in the same folder.
"""

import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "app.py")

REQUIRED = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "plotly": "plotly",
    "statsmodels": "statsmodels",
    "matplotlib": "matplotlib",
}


def python_exe():
    """IDLE uses pythonw.exe (no console). Use python.exe so errors are visible."""
    exe = sys.executable
    if exe.lower().endswith("pythonw.exe"):
        candidate = exe[:-len("pythonw.exe")] + "python.exe"
        if os.path.exists(candidate):
            return candidate
    return exe


def install_missing(py):
    missing = [pkg for mod, pkg in REQUIRED.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        print("All libraries are already installed.")
        return
    print("Installing:", ", ".join(missing), "(this can take a few minutes)...")
    subprocess.check_call([py, "-m", "pip", "install", *missing])


def skip_welcome_email():
    """Streamlit asks for an email on first run. Pre-answer it with a blank."""
    folder = os.path.join(os.path.expanduser("~"), ".streamlit")
    path = os.path.join(folder, "credentials.toml")
    if not os.path.exists(path):
        os.makedirs(folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write('[general]\nemail = ""\n')


def main():
    if not os.path.exists(APP):
        print("Could not find app.py. Put run.py and app.py in the same folder.")
        return
    py = python_exe()
    print("Using Python:", py)
    install_missing(py)
    skip_welcome_email()

    print("\nStarting MPG Studio... wait 10-20 seconds.")
    print("Open this link in your browser:  http://localhost:8501")
    print("To stop: Shell > Restart Shell (Ctrl+F6) in IDLE, or Ctrl+C in a terminal.\n")

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(
        [py, "-m", "streamlit", "run", APP,
         "--server.port", "8501",
         "--server.headless", "false",
         "--browser.gatherUsageStats", "false"],
        cwd=HERE, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    try:
        for line in proc.stdout:       # show Streamlit messages and any errors
            print(line, end="")
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        print("\nStopped.")


if __name__ == "__main__":
    main()