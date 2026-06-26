<h3 align="center">Bachelor project at Vrije Universiteit Amsterdam</h3>

# Interactive Movement Agent

A real-time movement improvisation system in which a humanoid avatar mirrors, contrasts, or responds to a user's body movements via webcam. The system implements a structured mode-switching algorithm grounded in improvisation theory, evaluated in a within-subjects user study as part of a Bachelor's thesis at Vrije Universiteit Amsterdam.

Built on top of [Constructing a Movement Improvisational Agent](https://github.com/zjgb/Movement-Agent) by Zoja Gobec (VU Amsterdam, 2025).

The Unity scene is maintained in a separate repository due to its size: [interactive-movement-agent-unity](https://github.com/letix1/interactive-movement-agent-unity)

---

## How it works

The system has three components running in parallel:

- **Python backend** — captures webcam frames, estimates body pose using MediaPipe BlazePose, computes joint angles, and applies the active behavioral mode (mirroring, reversed mirroring, or contrasting).
- **OSC bridge** — filters the joint angles and forwards them to Unity via OSC/UDP on port 9000.
- **Unity scene** — drives the Robot Kyle humanoid avatar skeleton from the incoming joint data in real time.

The researcher selects the behavioral mode and AUTO/RANDOM algorithm via a PyQt6 control interface.

---

## System requirements

- Python 3.11
- Miniforge (ARM-native conda) — required for Apple Silicon Macs
- Unity 2022.3+ (URP) — see the [Unity repository](https://github.com/letix1/interactive-movement-agent-unity)
- Webcam

---

## Installation

Clone the repository:

```bash
git clone https://github.com/letix1/interactive-movement-agent.git
cd interactive-movement-agent
```

This project requires Miniforge on Apple Silicon Macs. [Download Miniforge here](https://github.com/conda-forge/miniforge).

Create and activate a conda environment with Python 3.11:

```bash
conda create -n movement-agent python=3.11
conda activate movement-agent
```

Install pybullet via conda (no pip wheel available for Apple Silicon):

```bash
conda install -c conda-forge pybullet
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the system

**1. Open the Unity scene** and enter Play mode (see [Unity repository](https://github.com/letix1/interactive-movement-agent-unity)).

**2. Start the Python backend:**

```bash
python main.py
```

This launches the PyQt6 control interface, starts the webcam feed, and automatically starts the OSC bridge in the background. A 10-second countdown gives you time to step in front of the camera before tracking begins — press any key in the terminal to skip it.

**3. Select a mode** in the interface (Mirroring, Reversed Mirroring, Contrasting) or enable AUTO or RANDOM for the experimental/baseline algorithm.

> Make sure Unity is in Play mode before starting the Python backend, and that port 9000 (OSC) and port 5555 (ZMQ) are free. If port 5000 is occupied by a previous crashed instance, run `kill -9 $(lsof -ti :5000)` before starting.
