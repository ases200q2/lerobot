---
noteId: "e739ff90b09c11f08bd0898d85b76daa"
tags: []
---

# LeRobot Comprehensive Documentation

This directory contains comprehensive documentation for LeRobot, covering all aspects of robot learning from data collection to model deployment.

## 📚 Documentation Overview

### Core Workflows

1. **[Data Collection Guide](Data_Collection_Guide.md)** - Complete guide for collecting demonstration data
   - General data collection workflows
   - SpaceMouse teleoperation setup
   - Configuration files and best practices
   - Troubleshooting common issues

2. **[Data Replay Guide](Data_Replay_Guide.md)** - How to replay and analyze demonstration data
   - Basic and advanced replay methods
   - Custom replay scripts and tools
   - Data analysis and visualization
   - Quality assessment techniques

3. **[Model Training Guide](Model_Training_Guide.md)** - Training robot learning models
   - ACT (Action Chunking Transformer) training
   - Other policy training (Diffusion, TDMPC, SmolVLA)
   - Training configuration and optimization
   - Evaluation and testing procedures

4. **[Hugging Face Upload Guide](Hugging_Face_Upload_Guide.md)** - Sharing datasets and models
   - Dataset upload workflows
   - Model upload procedures
   - Community guidelines and best practices
   - Upload optimization and troubleshooting

### Environment-Specific Guides

5. **[Simulation Environments Guide](Simulation_Environments_Guide.md)** - Working with simulation
   - MuJoCo environment setup
   - Isaac Sim integration
   - Gym-HIL environments
   - Custom simulation development

6. **[Real-World Data Collection Guide](Real_World_Data_Collection_Guide.md)** - Physical robot setup
   - Supported robot platforms (ALOHA, SO-100/SO-101, Koch)
   - Hardware setup and safety considerations
   - Robot-specific configuration guides
   - Quality assurance and troubleshooting

## 🚀 Quick Start

### 1. Data Collection

```bash
# Basic data collection with SpaceMouse
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --teleop.type=spacemouse \
  --dataset.repo_id=${HF_USER}/my_dataset \
  --dataset.num_episodes=20
```

### 2. Model Training

```bash
# Train ACT model
lerobot-train \
  --dataset.repo_id=${HF_USER}/my_dataset \
  --policy.type=act \
  --output_dir=outputs/train/act_model \
  --steps=20000
```

### 3. Data Replay

```bash
# Replay demonstration data
lerobot-replay \
  --robot.type=so101_follower \
  --dataset.repo_id=${HF_USER}/my_dataset \
  --dataset.episode=0
```

### 4. Upload to Hub

```bash
# Upload dataset to Hugging Face
huggingface-cli upload ${HF_USER}/my_dataset ~/.cache/huggingface/lerobot/my_dataset --repo-type dataset
```

## 🎯 Use Cases

### Simulation-Based Learning

- **MuJoCo**: Physics-based manipulation tasks
- **Isaac Sim**: High-fidelity robotics simulation
- **Gym-HIL**: Human-in-the-loop training

### Real-World Robot Learning

- **ALOHA**: Bimanual manipulation tasks
- **SO-100/SO-101**: Mobile manipulation
- **Koch**: Desktop manipulation tasks

### Data Collection Methods

- **SpaceMouse**: 6-DOF intuitive control
- **Gamepad**: Accessible teleoperation
- **Keyboard**: Simple control interface
- **Custom Devices**: Integration with specialized hardware

## 🔧 Configuration

### Environment Setup

```bash
# Install LeRobot
pip install -e ".[all]"

# Setup Hugging Face authentication
huggingface-cli login --token ${HUGGINGFACE_TOKEN}

# Verify installation
python -c "import lerobot; print('LeRobot installed successfully')"
```

### Robot Configuration

```json
{
  "robot": {
    "type": "so101_follower",
    "port": "/dev/ttyACM0",
    "id": "my_robot",
    "cameras": {
      "front": {
        "type": "opencv",
        "index_or_path": 0,
        "width": 640,
        "height": 480,
        "fps": 30
      }
    }
  }
}
```

### Training Configuration

```json
{
  "policy": {
    "type": "act",
    "dim_model": 512,
    "chunk_size": 100,
    "use_vae": true
  },
  "training": {
    "batch_size": 32,
    "steps": 20000,
    "device": "cuda"
  }
}
```

## 📊 Supported Policies

| Policy        | Description                  | Best For                           |
| ------------- | ---------------------------- | ---------------------------------- |
| **ACT**       | Action Chunking Transformer  | Manipulation tasks, smooth control |
| **Diffusion** | Diffusion-based policy       | Diverse action generation          |
| **TDMPC**     | Temporal Difference MPC      | Model-based control                |
| **SmolVLA**   | Small Vision-Language-Action | Vision-language tasks              |

## 🤖 Supported Robots

| Robot             | Type     | Applications                         |
| ----------------- | -------- | ------------------------------------ |
| **ALOHA**         | Bimanual | Assembly, bimanual manipulation      |
| **SO-100/SO-101** | Mobile   | Household tasks, mobile manipulation |
| **Koch**          | Desktop  | Pick and place, desktop tasks        |
| **Custom**        | Various  | Integration with custom hardware     |

## 📈 Data Formats

### LeRobotDataset v3.0

- **File-based storage**: Efficient episode management
- **Relational metadata**: Rich episode and frame information
- **Hub-native streaming**: Direct consumption from Hugging Face Hub
- **Unified organization**: Consistent directory structure

### Supported Data Types

- **Actions**: Robot joint commands, end-effector poses
- **Observations**: Joint states, camera images, sensor data
- **Metadata**: Task descriptions, episode information, quality metrics

## 🔍 Quality Assurance

### Data Quality Metrics

- **Episode Length**: Consistent demonstration duration
- **Action Smoothness**: Low acceleration, natural movements
- **Camera Quality**: Proper lighting, focus, and framing
- **Safety Compliance**: No safety violations or dangerous actions

### Validation Procedures

- **Pre-collection**: Environment setup, device testing
- **During Collection**: Real-time monitoring, quality checks
- **Post-collection**: Dataset validation, quality assessment

## 🛡️ Safety Considerations

### Hardware Safety

- Emergency stop buttons
- Safety barriers and cages
- Warning signs and lights
- First aid and fire safety equipment

### Software Safety

- Collision detection systems
- Joint limit monitoring
- Velocity and acceleration limits
- Emergency stop software
- Safety monitoring scripts

## 📚 Additional Resources

### Official Documentation

- [LeRobot GitHub Repository](https://github.com/huggingface/lerobot)
- [Hugging Face Hub](https://huggingface.co/lerobot)
- [LeRobot Documentation](https://huggingface.co/docs/lerobot)

### Community Resources

- [LeRobot Discord](https://discord.gg/lerobot)
- [LeRobot Forum](https://github.com/huggingface/lerobot/discussions)
- [LeRobot Examples](https://github.com/huggingface/lerobot/tree/main/examples)

### Research Papers

- [LeRobot Paper](https://arxiv.org/abs/2406.14496)
- [ACT Paper](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy Paper](https://arxiv.org/abs/2303.04137)

## 🤝 Contributing

We welcome contributions to LeRobot! Please see our [Contributing Guide](https://github.com/huggingface/lerobot/blob/main/CONTRIBUTING.md) for details.

### How to Contribute

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

### Areas for Contribution

- New robot integrations
- Additional policy implementations
- Documentation improvements
- Bug fixes and optimizations
- Community examples and tutorials

## 📄 License

LeRobot is licensed under the Apache 2.0 License. See [LICENSE](https://github.com/huggingface/lerobot/blob/main/LICENSE) for details.

## 🙏 Acknowledgments

- The LeRobot team at Hugging Face
- The robotics research community
- Open-source contributors and maintainers
- Robot hardware manufacturers and developers

---

**Happy Robot Learning! 🤖✨**

For questions, issues, or contributions, please visit our [GitHub repository](https://github.com/huggingface/lerobot) or join our [Discord community](https://discord.gg/lerobot).
