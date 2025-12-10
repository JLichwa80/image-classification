"""Restart the Gradio app for the pneumonia image-classification project.

Usage:
    python restart_gradio.py
"""

import subprocess
import sys
import os

def kill_gradio_processes():
    """Kill existing Gradio processes by finding those running app.py (Windows)."""
    print("Stopping existing Gradio processes...")
    try:
        # Find and kill only processes running app.py, not this script
        result = subprocess.run(
            ["wmic", "process", "where", "commandline like '%app.py%'", "get", "processid"],
            capture_output=True,
            text=True,
            check=False
        )
        for line in result.stdout.strip().split('\n')[1:]:
            pid = line.strip()
            if pid and pid.isdigit():
                subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True, check=False)
    except Exception:
        pass

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "deployment", "gradio-app", "app.py")

    # Kill existing Gradio processes
    kill_gradio_processes()

    # Set environment variable to disable Gradio auto-reload
    os.environ["GRADIO_WATCH_DIRS"] = "false"

    print(f"Starting Gradio app: {app_path}")

    # Run the app
    subprocess.run([sys.executable, app_path])

if __name__ == "__main__":
    main()
