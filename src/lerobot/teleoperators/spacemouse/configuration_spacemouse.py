#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpaceMouseTeleopConfig(TeleoperatorConfig):
    """SpaceMouse teleoperator configuration."""

    # Control mode: 5dof (X,Y,Z,gripper) or 7dof (X,Y,Z,roll,pitch,yaw,gripper)
    mode: str = "5dof"
    # Movement sensitivity (multiplier for SpaceMouse input)
    sensitivity: float = 0.2
    # Deadzone to prevent drift (minimum input threshold)
    deadzone: float = 0.05
    # Whether to enable gripper control
    use_gripper: bool = True
    # Whether to use pyspacemouse library (True) or HIDAPI (False)
    use_pyspacemouse: bool = True
