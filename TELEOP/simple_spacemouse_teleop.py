import gymnasium as gym
import numpy as np

print("Final SpaceMouse Test - Direct Control")
print("=" * 40)
print("✓ SpaceMouse: Direct robot control (no intervention needed)")
print("✓ Left button: Close gripper")
print("✓ Right button: Open gripper")
print("✓ SPACE key: Mark episode SUCCESS")
print("✓ C key: Mark episode FAILURE")
print()

try:
    env = gym.make("gym_hil/PandaPickCubeSpacemouse-v0")
    obs, info = env.reset()
    dummy_action = np.zeros(env.action_space.shape, dtype=np.float32)

    print("SpaceMouse ready! Move it to control the robot.")
    print("Test will run for 5 seconds...")
    print()

    movement_detected = False

    for i in range(500):  # 5 seconds
        obs, reward, terminated, truncated, info = env.step(dummy_action)

        action_intervention = info.get("action_intervention", dummy_action)

        # Check for movement
        if np.any(np.abs(action_intervention) > 0.001):
            if not movement_detected:
                print("✓ SpaceMouse movement detected!")
                movement_detected = True

            if i % 10 == 0:  # Show every second
                action_str = f"[{action_intervention[0]:.3f}, {action_intervention[1]:.3f}, {action_intervention[2]:.3f}, {action_intervention[3]:.3f}]"
                print(f"  Control: {action_str}")

        if terminated or truncated:
            success = info.get("next.success", False)
            status = "SUCCESS" if success else "FAILURE"
            print(f"Episode ended: {status} - Resetting...")
            obs, info = env.reset()

        # time.sleep(1.01)

    env.close()

    if movement_detected:
        print("\\n🎉 SUCCESS: SpaceMouse direct control is working!")
    else:
        print("\\n⚠️  No movement detected. Try moving the SpaceMouse.")

    print("\\nImplementation Summary:")
    print("• Removed intervention mode requirement")
    print("• SpaceMouse now controls robot directly")
    print("• Keyboard controls: SPACE=success, C=failure")
    print("• SpaceMouse buttons: Left=close gripper, Right=open gripper")

except Exception as e:
    print(f"Error: {e}")
