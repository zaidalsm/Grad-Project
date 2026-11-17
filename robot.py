"""Robot connection. MOCK for now: commands are only logged, nothing moves.

app.py and the web page use only two things from this file:
    send_command(cmd)   and   status()
When the robot students are ready (Week 6 or later), replace the bodies of
these two functions with the real ROS 2 connection. Nothing else in the
project has to change.

Open decisions to agree with the robot students (see INTERFACE.md):
  - the command format and the speed units
  - the ROS topic names
"""
from datetime import datetime

MODE = "mock"      # change to "ros" when the real connection exists
VALID_COMMANDS = ("forward", "back", "left", "right", "stop")

# Idea for later: how each button could map to (linear m/s, angular rad/s).
# Do NOT use these numbers until the team has agreed the units and speeds.
# SPEEDS = {"forward": (0.1, 0.0), "back": (-0.1, 0.0),
#           "left": (0.0, 0.5), "right": (0.0, -0.5), "stop": (0.0, 0.0)}

_last = {"cmd": None, "time": None}


def send_command(cmd):
    """Send one movement command. Raises ValueError for an unknown command."""
    if cmd not in VALID_COMMANDS:
        raise ValueError(f"Unknown command: {cmd}")
    _last["cmd"] = cmd
    _last["time"] = datetime.now().strftime("%H:%M:%S")
    print(f"[robot:{MODE}] command: {cmd}")
    return {"ok": True, "cmd": cmd, "mode": MODE}


def status():
    """Connection state shown on the page. Mock mode is never 'connected'."""
    return {"connected": False, "mode": MODE,
            "last_command": _last["cmd"], "last_time": _last["time"]}