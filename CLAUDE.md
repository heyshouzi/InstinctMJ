# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Install dependencies
uv sync
uv pip install -e ../mjlab -e ../instinct_rl

# Format code
instinct-format

# Train a task
instinct-train Instinct-Parkour-Target-Amp-T2_v3-v0

# Play/evaluate a trained policy
instinct-play Instinct-Parkour-Target-Amp-T2_v3-Play-v0 --load-run <run_name>

# List available tasks
instinct-list-envs

# Verify task registration
python -c "from instinct_mj.tasks.registry import list_tasks"
```

## Architecture Overview

**InstinctMJ** is the mjlab-native port of InstinctLab, serving as the environment side of Project-Instinct for humanoid whole-body RL control.

### Ecosystem Integration
- **mjlab**: Physics simulation backend (Mujoco + mjlab wrapper)
- **mujoco_warp**: MuJoCo Python bindings
- **instinct_rl**: Training workflow (train/play/export)
- **InstinctMJ**: Task suite (locomotion, shadowing, perceptive, parkour)

### Key Components
- `src/instinct_mj/assets/`: Robot configurations (unitree_g1.py, booster_t2*.py)
- `src/instinct_mj/tasks/`: Task definitions with mjlab-style configs
- `src/instinct_mj/envs/mdp/`: MDP components (rewards, observations, actions, commands)
- `src/instinct_mj/managers/`: Reward and manager configurations
- `src/instinct_mj/scene/`: Scene setup for simulation
- `src/instinct_mj/sensors/`: Camera and raycaster sensors

### Task Registration Pattern
Tasks are registered in `__init__.py` files using `register_instinct_task()`:
```python
from instinct_mj.tasks.registry import register_instinct_task

register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-T2_v3-v0",
    env_cfg_factory=lambda: my_task_cfg(play=False),
    play_env_cfg_factory=lambda: my_task_cfg(play=True),
    instinct_rl_cfg_factory=MyRunnerCfg,
)
```

### Robot Asset Structure
- `unitree_g1.py`: G1 robot (G1_29DOF_CFG, G1_30DOF_CFG)
- `booster_t2.py`: T2 V11 robot (T2_31DOF_CFG)
- `booster_t2_v3.py`: T2 V3 robot (T2_v3_31DOF_CFG) - G1-style naming
- `booster_t2_v3_2.py`: T2 V3.2 robot (T2_v3_31DOF_CFG) - lower natural frequency (4 vs 10)

### PD Control Parameters
KP = armature × (2π × natural_freq)²
Where natural_freq is defined per robot in assets (e.g., NATURAL_FREQ = 10 for T2_v3, NATURAL_FREQ = 4 for T2_v3_2)

## AGENTS.md Guidelines

When working in this repository:
- **Scope**: Changes should stay within InstinctMJ unless user explicitly asks for cross-project modifications
- **Migration**: Follow mjlab-native patterns; do not add compat layers
- **Code style**: 2-space indentation, explicit type annotations, dataclass-based configs
- **instinct_rl**: Do not modify unless user explicitly requests; if a fix requires it, ask first
- **Sim-to-Sim**: Policy checkpoint joint ordering differs between simulators; do not mix checkpoints across different simulator setups

## Directory Structure

```
src/instinct_mj/
├── assets/          # Robot URDF/MJCF configs, PD parameters
├── actuators/       # Actuator implementations
├── envs/mdp/
│   ├── rewards/    # Reward functions
│   ├── observations/# Observation builders
│   ├── actions/     # Action interfaces
│   ├── commands/    # Command generators
│   └── events/      # Randomization, terrain events
├── managers/        # Manager term configurations
├── motion_reference/# Motion playback and reference systems
├── scene/           # Scene setup
├── sensors/         # Camera and raycaster sensors
├── tasks/           # Task definitions
│   ├── locomotion/  # Flat/rough terrain locomotion
│   ├── parkour/     # Parkour challenge tasks
│   └── registry.py  # Task registration
├── monitors/        # Logging and monitoring
└── scripts/         # Utilities (format, list_envs, multi_play)
```
