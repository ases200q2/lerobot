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

import logging
import time
from queue import Queue
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .configuration_spacemouse import SpaceMouseTeleopConfig

# Try to import gym-hil SpaceMouse controller
try:
    from gym_hil.wrappers.intervention_utils import (
        SpaceMouseControllerHIDAPI,
        SpaceMouseControllerPySpaceMouse,
    )

    GYM_HIL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import gym-hil SpaceMouse controller: {e}")
    GYM_HIL_AVAILABLE = False


class SpaceMouseTeleop(Teleoperator):
    """
    Teleoperator class to use SpaceMouse input for robot control.
    """

    config_class = SpaceMouseTeleopConfig
    name = "spacemouse"

    def __init__(self, config: SpaceMouseTeleopConfig):
        super().__init__(config)
        self.config = config
        self.controller = None
        self.misc_keys_queue = Queue()
        self.logs = {}

    @property
    def action_features(self) -> dict:
        """Return action features based on mode and gripper usage."""
        if self.config.mode == "7dof":
            if self.config.use_gripper:
                return {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": {
                        "delta_x": 0,
                        "delta_y": 1,
                        "delta_z": 2,
                        "delta_roll": 3,
                        "delta_pitch": 4,
                        "delta_yaw": 5,
                        "gripper": 6,
                    },
                }
            else:
                return {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": {
                        "delta_x": 0,
                        "delta_y": 1,
                        "delta_z": 2,
                        "delta_roll": 3,
                        "delta_pitch": 4,
                        "delta_yaw": 5,
                    },
                }
        else:  # 5dof mode
            if self.config.use_gripper:
                return {
                    "dtype": "float32",
                    "shape": (4,),
                    "names": {
                        "delta_x": 0,
                        "delta_y": 1,
                        "delta_z": 2,
                        "gripper": 3,
                    },
                }
            else:
                return {
                    "dtype": "float32",
                    "shape": (3,),
                    "names": {
                        "delta_x": 0,
                        "delta_y": 1,
                        "delta_z": 2,
                    },
                }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.controller is not None and hasattr(self.controller, "running") and self.controller.running

    @property
    def is_calibrated(self) -> bool:
        return True  # SpaceMouse doesn't require calibration

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "SpaceMouse is already connected. Do not run `connect()` twice."
            )

        if not GYM_HIL_AVAILABLE:
            raise ImportError(
                "gym-hil is not available. Please install gym-hil to use SpaceMouse teleoperator."
            )

        # Initialize SpaceMouse controller
        if self.config.use_pyspacemouse:
            try:
                self.controller = SpaceMouseControllerPySpaceMouse(
                    mode=self.config.mode,
                    sensitivity=self.config.sensitivity,
                    deadzone=self.config.deadzone,
                )
            except ImportError:
                logging.warning("pyspacemouse not available, falling back to HIDAPI")
                self.controller = SpaceMouseControllerHIDAPI(
                    mode=self.config.mode,
                    sensitivity=self.config.sensitivity,
                    deadzone=self.config.deadzone,
                )
        else:
            self.controller = SpaceMouseControllerHIDAPI(
                mode=self.config.mode,
                sensitivity=self.config.sensitivity,
                deadzone=self.config.deadzone,
            )

        self.controller.start()
        logging.info("SpaceMouse connected successfully")

    def calibrate(self) -> None:
        pass  # SpaceMouse doesn't require calibration

    def configure(self):
        pass

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "SpaceMouse is not connected. You need to run `connect()` before `get_action()`."
            )

        before_read_t = time.perf_counter()

        # Update controller to get fresh inputs
        self.controller.update()

        # Get movement deltas from the controller
        x, y, z, roll, pitch, yaw = self.controller.get_deltas()

        # Create action dictionary based on mode
        if self.config.mode == "7dof":
            action = {
                "delta_x": x,
                "delta_y": y,
                "delta_z": z,
                "delta_roll": roll,
                "delta_pitch": pitch,
                "delta_yaw": yaw,
            }
            if self.config.use_gripper:
                gripper_command = self.controller.gripper_command()
                if gripper_command == "open":
                    action["gripper"] = 2.0
                elif gripper_command == "close":
                    action["gripper"] = 0.0
                else:
                    action["gripper"] = 1.0
        else:  # 5dof mode
            action = {
                "delta_x": x,
                "delta_y": y,
                "delta_z": z,
            }
            if self.config.use_gripper:
                gripper_command = self.controller.gripper_command()
                if gripper_command == "open":
                    action["gripper"] = 2.0
                elif gripper_command == "close":
                    action["gripper"] = 0.0
                else:
                    action["gripper"] = 1.0

        # Check for episode control events
        episode_end_status = self.controller.get_episode_end_status()
        if episode_end_status is not None:
            if episode_end_status == "success":
                self.misc_keys_queue.put(TeleopEvents.SUCCESS)
            elif episode_end_status == "failure":
                self.misc_keys_queue.put(TeleopEvents.FAILURE)
            elif episode_end_status == "rerecord_episode":
                self.misc_keys_queue.put(TeleopEvents.RERECORD_EPISODE)

        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "SpaceMouse is not connected. You need to run `connect()` before `disconnect()`."
            )
        if self.controller is not None:
            self.controller.stop()
            self.controller = None
        logging.info("SpaceMouse disconnected")

    def get_teleop_events(self) -> dict[str, Any]:
        """Get teleoperator events from the misc_keys_queue."""
        events = {}
        try:
            while not self.misc_keys_queue.empty():
                event = self.misc_keys_queue.get_nowait()
                if event == TeleopEvents.SUCCESS:
                    events["success"] = True
                elif event == TeleopEvents.FAILURE:
                    events["failure"] = True
                elif event == TeleopEvents.RERECORD_EPISODE:
                    events["rerecord_episode"] = True
        except Exception as e:
            logging.debug(f"Error processing misc keys queue: {e}")  # nosec B110
        return events
