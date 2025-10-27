# Troubleshooting Guide

Comprehensive guide for resolving common issues with the SpaceMouse IL pipeline.

## Table of Contents

- [Data Collection Issues](#data-collection-issues)
- [Training Issues](#training-issues)
- [Evaluation Issues](#evaluation-issues)
- [Environment Issues](#environment-issues)
- [Hardware Issues](#hardware-issues)
- [HuggingFace Hub Issues](#huggingface-hub-issues)
- [Performance Issues](#performance-issues)
- [Data Quality Issues](#data-quality-issues)

---

## Data Collection Issues

### SpaceMouse Not Detected

**Symptom:** "SpaceMouse device NOT detected" warning during data collection

**Causes & Solutions:**

1. **SpaceMouse not connected**
   ```bash
   # Check USB connection
   lsusb | grep -i 3dconnexion

   # If not found, reconnect USB cable and try again
   ```

2. **hidapi library not installed**
   ```bash
   # Install hidapi
   pip install hidapi

   # Verify installation
   python -c "import hid; print(hid.enumerate())"
   ```

3. **Permission issues (Linux)**
   ```bash
   # Create udev rule for SpaceMouse
   sudo nano /etc/udev/rules.d/99-spacemouse.rules

   # Add this line (replace XXXX with your device ID from lsusb):
   SUBSYSTEM=="usb", ATTR{idVendor}=="256f", MODE="0666"

   # Reload udev rules
   sudo udevadm control --reload-rules
   sudo udevadm trigger

   # Reconnect SpaceMouse
   ```

4. **Driver issues (macOS)**
   ```bash
   # Download and install 3Dconnexion drivers
   # https://3dconnexion.com/us/drivers/
   ```

### Episode Recording Fails

**Symptom:** Episodes fail to save or recording crashes

**Solutions:**

1. **Check disk space**
   ```bash
   df -h ~/.cache/huggingface/lerobot/

   # If low, free up space or change cache directory:
   export HF_DATASETS_CACHE="/path/to/larger/drive"
   ```

2. **Check HuggingFace credentials**
   ```bash
   huggingface-cli whoami

   # If not logged in:
   huggingface-cli login
   ```

3. **Verify repository exists**
   ```bash
   # Check if repo exists
   huggingface-cli repo info username/dataset --repo-type dataset

   # Create repo if needed
   huggingface-cli repo create username/dataset --type dataset
   ```

### Camera Issues

**Symptom:** No camera images or black frames

**Solutions:**

1. **Test camera access**
   ```python
   import cv2
   cap = cv2.VideoCapture(0)
   print("Camera opened:", cap.isOpened())
   ret, frame = cap.read()
   print("Frame shape:", frame.shape if ret else "None")
   cap.release()
   ```

2. **Check gym_hil installation**
   ```bash
   pip install -e ".[hilserl]" --upgrade
   ```

3. **Verify MuJoCo rendering**
   ```bash
   # Test MuJoCo rendering
   python -c "import mujoco; import glfw; print('MuJoCo and GLFW OK')"
   ```

### Slow Data Collection

**Symptom:** FPS is much lower than configured 10 FPS

**Solutions:**

1. **Disable camera display**
   ```json
   // In configs/collect_data_config.json
   "display_cameras": false
   ```

2. **Use headless mode**
   ```bash
   # Set environment variable
   export MUJOCO_GL=osmesa  # Or egl for GPU
   ```

3. **Reduce image resolution**
   ```json
   "resize_size": [64, 64]  // Instead of [128, 128]
   ```

---

## Training Issues

### CUDA Out of Memory

**Symptom:** `RuntimeError: CUDA out of memory`

**Solutions:**

1. **Reduce batch size**
   ```bash
   ./scripts/run_training.sh username/dataset 100000 16  # Batch size 16
   # Or even smaller:
   ./scripts/run_training.sh username/dataset 100000 8   # Batch size 8
   ```

2. **Use gradient accumulation**
   ```bash
   python scripts/train_policy.py \
       --dataset_id username/dataset \
       --batch_size 8 \
       --gradient_accumulation_steps 4  # Effective batch size: 8*4=32
   ```

3. **Clear GPU cache**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

4. **Check GPU memory usage**
   ```bash
   nvidia-smi --loop=1

   # Look for other processes using GPU
   # Kill unnecessary processes if needed
   ```

### Dataset Not Found

**Symptom:** `DatasetNotFoundError` or HTTP 404 error

**Solutions:**

1. **Verify dataset exists**
   ```bash
   huggingface-cli repo info username/dataset --repo-type dataset
   ```

2. **Check authentication**
   ```bash
   huggingface-cli whoami
   # Should show your username
   ```

3. **Verify dataset ID format**
   ```bash
   # Correct format: username/dataset-name
   # NOT: username/models/dataset-name
   # NOT: dataset-name (missing username)
   ```

4. **Check dataset is public or you have access**
   - Private datasets require authentication
   - Verify access on HuggingFace Hub website

### Training Diverges or NaN Loss

**Symptom:** Loss becomes NaN or training doesn't improve

**Solutions:**

1. **Reduce learning rate**
   ```bash
   python scripts/train_policy.py \
       --dataset_id username/dataset \
       --config configs/train_act_config.json

   # Edit config to reduce lr:
   "lr": 5e-6  # Instead of 1e-5
   ```

2. **Check data quality**
   ```bash
   # Visualize dataset to ensure quality
   lerobot-dataset-viz --repo-id username/dataset
   ```

3. **Enable gradient clipping** (already enabled by default)
   ```json
   "grad_clip_norm": 10  // Prevents exploding gradients
   ```

4. **Check for corrupted episodes**
   ```python
   from lerobot.datasets.lerobot_dataset import LeRobotDataset
   dataset = LeRobotDataset("username/dataset")
   print(f"Num episodes: {dataset.num_episodes}")
   print(f"Num frames: {dataset.num_frames}")

   # Check for anomalies
   for ep_idx in range(dataset.num_episodes):
       ep = dataset.get_episode(ep_idx)
       if len(ep) == 0:
           print(f"Empty episode: {ep_idx}")
   ```

### Training Too Slow

**Symptom:** Training takes much longer than expected

**Solutions:**

1. **Check data loading**
   ```bash
   # Ensure dataset is cached locally
   ls -lh ~/.cache/huggingface/lerobot/username_dataset/

   # Pre-download if needed
   python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; \
              LeRobotDataset('username/dataset')"
   ```

2. **Use faster video backend**
   ```bash
   # Already set in training config:
   --dataset.video_backend=pyav  # Faster than opencv
   ```

3. **Increase num_workers**
   ```bash
   python scripts/train_policy.py \
       --dataset_id username/dataset \
       --num_workers 4  # Parallel data loading
   ```

4. **Check GPU utilization**
   ```bash
   nvidia-smi dmon -s u
   # GPU utilization should be >80%
   # If low, data loading is bottleneck
   ```

---

## Evaluation Issues

### Policy Fails to Load

**Symptom:** `FileNotFoundError` or `ModelNotFoundError`

**Solutions:**

1. **Check policy path**
   ```bash
   # For HuggingFace Hub:
   huggingface-cli repo info username/model_name

   # For local path:
   ls outputs/train/panda_spacemouse_act_*/checkpoints/last/pretrained_model/
   ```

2. **Verify checkpoint structure**
   ```bash
   # Should contain:
   # - config.json
   # - pytorch_model.bin (or model.safetensors)
   ls -la path/to/checkpoint/
   ```

3. **Check dataset metadata availability**
   ```bash
   # Policy needs dataset metadata for configuration
   python scripts/eval_policy.py \
       --policy_path username/model \
       --dataset_id username/dataset  # Provide dataset explicitly
   ```

### Poor Evaluation Performance

**Symptom:** Success rate < 30% despite good training loss

**Solutions:**

1. **Verify environment matches training**
   - Same observation preprocessing
   - Same FPS (10 Hz)
   - Same action space

2. **Check observation transformation**
   ```python
   # Debug observation shape
   obs, _ = env.reset()
   print("Raw obs keys:", obs.keys())

   wrapped_obs = wrapper._transform_obs(obs)
   print("Wrapped obs keys:", wrapped_obs.keys())
   print("Image shape:", wrapped_obs['observation.images.front'].shape)
   print("State shape:", wrapped_obs['observation.state'].shape)
   ```

3. **Collect more data**
   ```bash
   # 30 episodes may not be enough
   # Collect 50-100 episodes for better performance
   ./scripts/run_data_collection.sh 100
   ```

4. **Train for more steps**
   ```bash
   # Try 200k steps instead of 100k
   ./scripts/run_training.sh username/dataset 200000
   ```

### GUI Not Showing

**Symptom:** Evaluation runs but no visualization windows appear

**Solutions:**

1. **Check OpenCV backend**
   ```python
   import cv2
   print("OpenCV version:", cv2.__version__)
   print("GUI backend:", cv2.getBuildInformation())

   # Test window creation
   cv2.namedWindow("test")
   cv2.waitKey(1)
   cv2.destroyAllWindows()
   ```

2. **X11 forwarding (if SSH)**
   ```bash
   # SSH with X11 forwarding
   ssh -X user@host

   # Verify DISPLAY is set
   echo $DISPLAY
   ```

3. **Use headless mode**
   ```bash
   # If GUI not needed
   python scripts/eval_policy.py \
       --policy_path username/model \
       --no-gui
   ```

4. **MacOS specific**
   ```bash
   # Install XQuartz for GUI support
   brew install --cask xquartz
   ```

---

## Environment Issues

### gym_hil Not Installed

**Symptom:** `ModuleNotFoundError: No module named 'gym_hil'`

**Solutions:**

```bash
# Install gym_hil via hilserl extras
pip install -e ".[hilserl]"

# Verify installation
python -c "import gym_hil; print(gym_hil.__version__)"
```

### MuJoCo Issues

**Symptom:** `MuJoCo not found` or rendering errors

**Solutions:**

1. **Install MuJoCo**
   ```bash
   pip install mujoco

   # Verify installation
   python -c "import mujoco; print(mujoco.__version__)"
   ```

2. **Install rendering dependencies**
   ```bash
   # Linux
   sudo apt-get install libgl1-mesa-glx libglew-dev patchelf

   # macOS
   brew install glew
   ```

3. **Set MuJoCo rendering backend**
   ```bash
   export MUJOCO_GL=glfw  # Default, requires display
   # Or:
   export MUJOCO_GL=egl   # GPU-accelerated headless
   # Or:
   export MUJOCO_GL=osmesa  # CPU-based headless
   ```

### Environment Hangs or Freezes

**Symptom:** Environment creation or reset hangs indefinitely

**Solutions:**

1. **Kill stuck processes**
   ```bash
   # Find Python processes
   ps aux | grep python

   # Kill if necessary
   kill -9 <PID>
   ```

2. **Clear MuJoCo lock files**
   ```bash
   rm -rf /tmp/mujoco_*
   ```

3. **Restart from clean state**
   ```bash
   # Exit Python completely
   # Start fresh terminal session
   ```

---

## Hardware Issues

### GPU Not Detected

**Symptom:** Training/evaluation uses CPU instead of GPU

**Solutions:**

1. **Check CUDA installation**
   ```bash
   nvidia-smi
   # Should show GPU info

   nvcc --version
   # Should show CUDA version
   ```

2. **Check PyTorch CUDA**
   ```python
   import torch
   print("CUDA available:", torch.cuda.is_available())
   print("CUDA version:", torch.version.cuda)
   print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
   ```

3. **Reinstall PyTorch with CUDA**
   ```bash
   # Uninstall current PyTorch
   pip uninstall torch torchvision

   # Install with CUDA support (check https://pytorch.org for correct command)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Insufficient GPU Memory

**Symptom:** GPU has < 8 GB memory

**Solutions:**

1. **Use smaller batch size** (see Training Issues above)

2. **Use mixed precision training**
   ```bash
   # Add to training command
   --training.use_amp=true
   ```

3. **Reduce model size**
   ```json
   // In configs/train_act_config.json
   "dim_model": 128,  // Instead of 256
   "n_encoder_layers": 2,  // Instead of 4
   ```

---

## HuggingFace Hub Issues

### Authentication Failed

**Symptom:** `401 Unauthorized` or authentication errors

**Solutions:**

1. **Login to HuggingFace**
   ```bash
   huggingface-cli login
   # Enter your token from https://huggingface.co/settings/tokens
   ```

2. **Set token as environment variable**
   ```bash
   export HF_TOKEN=hf_...
   ```

3. **Check token permissions**
   - Token must have **write** access for uploading
   - Regenerate token if needed

### Upload Fails

**Symptom:** Dataset or model upload times out or fails

**Solutions:**

1. **Check network connection**
   ```bash
   ping huggingface.co
   ```

2. **Upload manually**
   ```bash
   huggingface-cli upload username/dataset \
       ~/.cache/huggingface/lerobot/dataset_name \
       --repo-type dataset
   ```

3. **Use git-lfs for large files**
   ```bash
   # Install git-lfs
   sudo apt-get install git-lfs  # Linux
   brew install git-lfs          # macOS

   git lfs install
   ```

4. **Retry with resume**
   ```bash
   # HuggingFace CLI supports resume for interrupted uploads
   huggingface-cli upload ... --resume
   ```

---

## Performance Issues

### Slow Evaluation

**Symptom:** Evaluation much slower than training FPS

**Solutions:**

1. **Disable visualization**
   ```bash
   python scripts/eval_policy.py --policy_path username/model --no-gui
   ```

2. **Check FPS timing**
   ```bash
   # Ensure FPS matches training (10 Hz)
   python scripts/eval_policy.py --policy_path username/model --fps 10
   ```

3. **Profile code**
   ```python
   import time
   start = time.time()
   # ... code to profile ...
   print(f"Time: {time.time() - start:.3f}s")
   ```

### High Memory Usage

**Symptom:** System runs out of RAM

**Solutions:**

1. **Reduce batch size during data loading**

2. **Close unused applications**

3. **Monitor memory usage**
   ```bash
   htop  # Or 'top' on macOS
   ```

4. **Use memory-efficient dataset loading**
   ```python
   # Don't load entire dataset into memory
   # Use streaming mode if available
   ```

---

## Data Quality Issues

### Inconsistent Demonstrations

**Symptom:** Policy learns poorly despite many demonstrations

**Solutions:**

1. **Review demonstrations**
   ```bash
   lerobot-dataset-viz --repo-id username/dataset
   ```

2. **Remove failed episodes**
   - Delete episodes marked as failures
   - Keep only high-quality success demonstrations

3. **Consistent task execution**
   - Use similar trajectories
   - Start from similar initial positions
   - Avoid unnecessary movements

### Poor Generalization

**Symptom:** Policy works on some episodes but not others

**Solutions:**

1. **Increase data diversity**
   - Vary initial object positions
   - Collect demonstrations from different angles
   - Use different grasping strategies

2. **Collect more data**
   - 30 episodes → 50-100 episodes
   - More data = better generalization

3. **Data augmentation** (advanced)
   - Add image augmentation during training
   - Perturb initial states

---

## Getting Help

If you've tried the solutions above and still have issues:

1. **Check LeRobot Documentation**
   - https://github.com/huggingface/lerobot
   - https://huggingface.co/docs/lerobot

2. **Search GitHub Issues**
   - https://github.com/huggingface/lerobot/issues

3. **Ask on Discord**
   - LeRobot Discord: https://discord.com/invite/s3KuuzsPFb

4. **Create GitHub Issue**
   - Provide detailed error messages
   - Include system information (OS, GPU, Python version)
   - Share relevant code snippets

---

**Happy Debugging! 🔧🤖**
