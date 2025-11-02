# integration.py
from module1 import all_theta_init  # Import the collected joint angles from module1
from module2 import HumanoidWalkEnv
import time

# Ensure URDF is generated (run urdf.py if needed to create the URDF file)
# from urdf import make_human  # Uncomment if you need to generate URDF dynamically

# Initialize the environment
env = HumanoidWalkEnv(render=True)  # Set render=False for headless mode

# Loop through all initial poses from Module 1
for name, initial_pose in all_theta_init.items():
    print(f"\nSimulating with initial pose from image: {name}")
    print(f"Initial joint angles: {initial_pose}")
    
    # Reset the environment with the initial pose
    obs, _ = env.reset(initial_pose=initial_pose)
    
    # Run a short simulation loop (e.g., 1000 steps or until done)
    for t in range(1000):
        action = env.action_space.sample()  # Random action for testing
        obs, reward, done, _, info = env.step(action)
        print(f"Step {t:03d} | Reward={reward:.3f} | fwd_vel={info['forward_vel']:.3f}")
        time.sleep(1/240.0)  # Match simulation FPS
        if done:
            print("Humanoid fell down, resetting with same pose...")
            obs, _ = env.reset(initial_pose=initial_pose)  # Re-use the pose
            break  # Or continue to next pose

# Close the environment after all simulations
env.close()
print("All simulations completed.")