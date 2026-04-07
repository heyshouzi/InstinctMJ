"""Register Instinct MJ parkour T2 tasks."""

# Copyright (c) 2022-2025, The Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from instinct_mj.tasks.registry import register_instinct_task

from .agents.instinct_rl_amp_cfg import T2ParkourPPORunnerCfg, T2_v3ParkourPPORunnerCfg
from .t2_parkour_target_amp_cfg import instinct_t2_parkour_amp_final_cfg
from .t2_v3_parkour_target_amp_cfg import instinct_t2_v3_parkour_amp_final_cfg

register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-T2-v0",
    env_cfg_factory=lambda: instinct_t2_parkour_amp_final_cfg(play=False),
    play_env_cfg_factory=lambda: instinct_t2_parkour_amp_final_cfg(play=True),
    instinct_rl_cfg_factory=T2ParkourPPORunnerCfg,
)


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-T2-Play-v0",
    env_cfg_factory=lambda: instinct_t2_parkour_amp_final_cfg(play=True),
    play_env_cfg_factory=lambda: instinct_t2_parkour_amp_final_cfg(play=True),
    instinct_rl_cfg_factory=T2ParkourPPORunnerCfg,
)

# T2 v3 task registrations
register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-T2_v3-v0",
    env_cfg_factory=lambda: instinct_t2_v3_parkour_amp_final_cfg(play=False),
    play_env_cfg_factory=lambda: instinct_t2_v3_parkour_amp_final_cfg(play=True),
    instinct_rl_cfg_factory=T2_v3ParkourPPORunnerCfg,
)


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-T2_v3-Play-v0",
    env_cfg_factory=lambda: instinct_t2_v3_parkour_amp_final_cfg(play=True),
    play_env_cfg_factory=lambda: instinct_t2_v3_parkour_amp_final_cfg(play=True),
    instinct_rl_cfg_factory=T2_v3ParkourPPORunnerCfg,
)
