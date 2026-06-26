"""
OSC Bridge — forwards joint angles from Python to Unity.

The main app publishes a pickled dictionary over ZMQ (tcp port 5555)
every frame. This script subscribes to that stream, extracts the 14 joint
rotations Unity needs, and sends them as a single OSC message (/joints, 42 floats)
to localhost:9000 where the Unity extOSC receiver is listening.

Joint order in the OSC message (each joint = 3 floats: tz, tx, ty in radians):
      [0–2] hips
      [3–5] left_hip
      [6–8] right_hip
     [9–11] left_knee
    [12–14] right_knee
    [15–17] left_ankle
    [18–20] right_ankle
    [21–23] neck
    [24–26] left_shoulder
    [27–29] right_shoulder
    [30–32] left_elbow
    [33–35] right_elbow
    [36–38] left_wrist
    [39–41] right_wrist
    [42–44] hip_position (avg ankle x, y, z — for vertical displacement)

The bridge is started automatically as a daemon thread by og_main.py.
It can also be run standalone for testing:
    python osc_bridge.py
"""

import pickle
import sys
import time
import zmq
import numpy as np
from pythonosc.udp_client import SimpleUDPClient
from ComputerVisionModules.oneEuroFilter import OneEuroFilter

# Configuration
ZMQ_ADDRESS = "tcp://localhost:5555"   # must match og_main.py's zmq.PAIR bind
OSC_IP      = "127.0.0.1"              # Unity IP
OSC_PORT    = 9000                     # must match JointReceiver.cs
OSC_ADDRESS = "/joints"                # OSC address pattern Unity binds to

FILTER_MIN_CUTOFF = 1.0     # Hz; raise for more smoothing, lower for less lag
FILTER_BETA       = 0.007   # raise to track fast motion more tightly

# Legs need less smoothing: single-camera depth is noisy for legs,
# but aggressive filtering makes them look rigid/laggy.
LEG_MIN_CUTOFF    = 1.7     # higher cutoff = less smoothing = more responsive
LEG_BETA          = 0.01    # faster speed tracking

# The joints in the order Unity expects them (matches JointReceiver.cs).
# Each entry is the key prefix in the angles dict; "_angles" is appended.
JOINT_ORDER = [
    "hips",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

# One filter per OSC float: 14 joints × 3 axes + 3 hip_position = 45 filters
_angle_keys = []
for _joint in JOINT_ORDER:
    for _axis in ['tz', 'tx', 'ty']:
        _angle_keys.append(f'{_joint}_{_axis}')

_position_keys = []
for _axis in ['x', 'y', 'z']:
    _position_keys.append(f'hip_pos_{_axis}')

_filter_keys = _angle_keys + _position_keys

# Joints that use the less-aggressive leg filter
_leg_joints = {'left_hip', 'right_hip', 'left_knee', 'right_knee',
               'left_ankle', 'right_ankle'}

one_euro_filters = {}
for k in _angle_keys:
    joint_name = k.rsplit('_', 1)[0]   # e.g. "left_knee_tz" → "left_knee"
    # strip axis suffix: last part after final '_' is tz/tx/ty
    parts = k.split('_')
    joint_name = '_'.join(parts[:-1])   # "left_knee"
    if joint_name in _leg_joints:
        one_euro_filters[k] = OneEuroFilter(
            min_cutoff=LEG_MIN_CUTOFF, beta=LEG_BETA, angular=True
        )
    else:
        one_euro_filters[k] = OneEuroFilter(
            min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA, angular=True
        )
for k in _position_keys:
    one_euro_filters[k] = OneEuroFilter(
        min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA, angular=False
    )


def main():
    # ZMQ subscriber
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PAIR)
    zmq_socket.connect(ZMQ_ADDRESS)
    print(f"[bridge] Connected to ZMQ at {ZMQ_ADDRESS}")

    # OSC sender
    osc_client = SimpleUDPClient(OSC_IP, OSC_PORT)
    print(f"[bridge] Sending OSC to {OSC_IP}:{OSC_PORT} on {OSC_ADDRESS}")
    print(f"[bridge] 1€ filter active — min_cutoff={FILTER_MIN_CUTOFF} beta={FILTER_BETA}")

    frame_count = 0

    while True:
        try:
            raw = zmq_socket.recv()
            angles_dict = pickle.loads(raw)
        except KeyboardInterrupt:
            print("\n[bridge] Interrupted — shutting down.")
            break
        except Exception as e:
            print(f"[bridge] ZMQ receive error: {e}")
            continue

        # Build the flat float list for Unity
        osc_values = []
        for joint in JOINT_ORDER:
            key = joint + "_angles"
            if key in angles_dict:
                angles = angles_dict[key]
                osc_values.extend([float(angles[0]), float(angles[1]), float(angles[2])])
            else:
                osc_values.extend([0.0, 0.0, 0.0])

        # Hip position (3 extra floats for vertical displacement)
        hip_pos = angles_dict.get('hip_position', np.array([0.0, 0.0, 0.0]))
        osc_values.extend([float(hip_pos[0]), float(hip_pos[1]), float(hip_pos[2])])

        # Apply 1€ filter to every float
        now = time.time()
        filtered_values = [
            one_euro_filters[_filter_keys[i]](osc_values[i], now)
            for i in range(len(osc_values))
        ]

        # Send to Unity via OSC
        osc_client.send_message(OSC_ADDRESS, filtered_values)

        frame_count += 1
        if frame_count % 60 == 0:
            print(f"[bridge] Forwarded {frame_count} frames")


def start_in_background():
    """Launch the bridge loop in a daemon thread (called by og_main.py)."""
    import threading
    t = threading.Thread(target=main, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    main()