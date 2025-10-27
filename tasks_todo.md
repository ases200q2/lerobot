---
noteId: "5abbb150b0b311f08bd0898d85b76daa"
tags: []
---

# URDF Robot Model Conversion: Isaac Sim to LeRobot

## Overview

This document provides guidance on converting URDF robot models between Isaac Sim and LeRobot, based on analysis of the LeRobot codebase and identification of existing conversion patterns.

## LeRobot URDF Integration

### Current URDF Usage in LeRobot

LeRobot has solid URDF support through the following components:

1. **RobotKinematics Class** (`/home/cihan/lerobot/src/lerobot/model/kinematics.py`)
   - Uses the `placo` library for forward and inverse kinematics
   - Loads URDF files for robot model representation
   - Provides FK/IK calculations for robot control

2. **Configuration System**
   - URDF paths are configured via `urdf_path` parameter in robot configs
   - Examples found in SO100/SO101 configurations: `"./SO101/so101_new_calib.urdf"`
   - Environment configs support URDF via `InverseKinematicsConfig`

3. **Simulation Support**
   - LeRobot integrates with `gym-hil` for MuJoCo-based simulation
   - Currently supports Franka Panda robot simulation
   - Uses URDF for robot model representation in simulation

### Key Files for URDF Handling

- `/home/cihan/lerobot/src/lerobot/model/kinematics.py` - Core kinematics using placo
- `/home/cihan/lerobot/src/lerobot/scripts/lerobot_find_joint_limits.py` - Joint limits discovery
- `/home/cihan/lerobot/src/lerobot/envs/configs.py` - Environment configuration including URDF paths
- Robot configuration files (e.g., `config_so100_follower.py`)

## Conversion Requirements: Isaac Sim to LeRobot

### 1. URDF File Compatibility

**Isaac Sim Export Requirements:**

- Export robot models as standard URDF files
- Ensure joint names and link names match LeRobot conventions
- Include proper material properties and visual meshes
- Export collision geometries for physics simulation

**LeRobot Expectations:**

- Standard URDF format with joint limits
- Properly defined end-effector frame (e.g., "gripper_frame_link")
- Joint names that match motor configuration

### 2. Asset Pipeline Recommendations

**From Isaac Sim to LeRobot:**

1. **Export from Isaac Sim:**

   ```bash
   # Isaac Sim Python script to export URDF
   import omni.kit.commands
   omni.kit.commands.execute("URDFExportCommand",
       usd_path="/path/to/robot.usd",
       urdf_path="/path/to/robot.urdf",
       export_collision=True,
       export_visual=True)
   ```

2. **URDF Processing:**
   - Verify joint names match LeRobot motor conventions
   - Add missing joint limits if needed
   - Ensure end-effector frame is properly defined

3. **LeRobot Configuration:**
   ```python
   # Example robot config for imported URDF
   @RobotConfig.register_subclass("isaac_robot")
   @dataclass
   class IsaacRobotConfig(RobotConfig):
       urdf_path: str = "./robots/isaac_robot.urdf"
       target_frame_name: str = "gripper_frame_link"
       # ... other configuration parameters
   ```

### 3. Integration with Existing Tools

**Using placo for Kinematics:**
LeRobot uses placo library for kinematics calculations. Ensure exported URDF is compatible:

- Joint types: revolute, prismatic, fixed
- Proper axis definitions
- Realistic joint limits

**Simulation Integration:**
For simulation use with gym-hil:

- Convert URDF to MuJoCo XML format
- Use existing MuJoCo integration patterns from Franka Panda example

## Practical Conversion Workflow

### Step 1: Export from Isaac Sim

1. Design/create robot in Isaac Sim
2. Ensure proper joint configuration and materials
3. Export as URDF with collision and visual meshes

### Step 2: URDF Validation

1. Validate URDF syntax using `check_urdf` tool
2. Verify joint names and limits
3. Ensure end-effector frame is properly defined

### Step 3: LeRobot Integration

1. Create robot configuration class
2. Set up URDF path in configuration
3. Define motor mappings and calibration parameters
4. Test with `lerobot-find-joint-limits` script

### Step 4: Testing

1. Test kinematics with RobotKinematics class
2. Validate forward/inverse kinematics
3. Test in simulation if needed

## Existing Conversion Patterns

### Dataset Format Conversion

LeRobot already has robust conversion tools, as seen in:

- `/home/cihan/lerobot/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py`
- Shows pattern: load → transform → save → validate

### Robot Configuration Pattern

Follow existing robot config patterns:

- Inherit from `RobotConfig` base class
- Use `@RobotConfig.register_subclass()` decorator
- Define urdf_path and target_frame_name parameters

## Recommended Tools and Libraries

### For Isaac Sim Side:

- Isaac Sim built-in URDF export
- Custom Python scripts for batch export
- Material and texture conversion utilities

### For LeRobot Side:

- **placo**: Kinematics calculations (already integrated)
- **gym-hil**: Simulation environment (MuJoCo-based)
- **lerobot-find-joint-limits**: Joint discovery tool

### Validation Tools:

- `check_urdf`: URDF syntax validation
- `urdf_to_graphviz`: Visualize robot structure
- LeRobot's own validation scripts

## Example: Franka Panda Integration

LeRobot already demonstrates this pattern with Franka Panda in gym-hil:

```python
# From docs/source/hilserl_sim.mdx
{
  "env": {
    "type": "gym_manipulator",
    "name": "gym_hil",
    "task": "PandaPickCubeGamepad-v0",
    "processor": {
      "inverse_kinematics": {
        "urdf_path": "path/to/panda.urdf",
        "target_frame_name": "gripper_frame_link"
      }
    }
  }
}
```

## Challenges and Solutions

### Challenge 1: Joint Name Mismatch

**Solution:** Create mapping dictionary in robot configuration

```python
joint_name_mapping = {
    "isaac_joint_1": "lerobot_motor_1",
    # ... other mappings
}
```

### Challenge 2: Different Units

**Solution:** LeRobot handles units via `use_degrees` parameter in robot configs

### Challenge 3: Material Properties

**Solution:** Convert Isaac Sim materials to standard URDF materials during export

## Future Enhancement Opportunities

1. **Automated Conversion Script:** Create a LeRobot script that automatically processes Isaac Sim exports
2. **Material Library:** Develop a standard material conversion library
3. **Validation Suite:** Extend existing validation tools to check Isaac Sim compatibility
4. **Asset Management:** Create asset pipeline for managing converted robot models

## Conclusion

LeRobot provides a solid foundation for URDF-based robot models with:

- Built-in kinematics support via placo
- Configuration-driven architecture
- Simulation integration through gym-hil
- Existing conversion patterns and tools

The main integration work involves:

1. Proper URDF export from Isaac Sim
2. Configuration setup in LeRobot
3. Validation and testing of the converted model

The existing LeRobot infrastructure makes this integration straightforward for most standard robot models.
