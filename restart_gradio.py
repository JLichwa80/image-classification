"""Restart the Gradio app for the pneumonia image-classification project.

Usage:
    python restart_gradio.py
"""

import subprocess
import sys
import os

def kill_python_processes():
    """Kill existing Python processes (Windows)."""
    print("Stopping existing Python/Gradio processes...")
    try:
        subprocess.run(
            ["taskkill", "/F", "/IM", "python.exe"],
            capture_output=True,
            check=False
        )
    except Exception:
        pass

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "deployment", "gradio-app", "app.py")

    # Kill existing processes
    kill_python_processes()

    # Set environment variable to disable Gradio auto-reload
    os.environ["GRADIO_WATCH_DIRS"] = "false"

    print(f"Starting Gradio app: {app_path}")

    # Run the app
    subprocess.run([sys.executable, app_path])

if __name__ == "__main__":
    main()
