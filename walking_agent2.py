import gym
env = gym.make("LunarLander-v2", render_mode="human")
env.action_space.seed(42)

observation, info = env.reset(seed=42)

def theta_omega_policy(observation):
    theta, w = observation[2:4]
    if abs(theta) < 0.03:
        return 0 if w < 0 else 1
    else:
        return 0 if theta < 0 else 1

for _ in range(1000):
    env.action_space = theta_omega_policy(observation)
    observation, reward, terminated, truncated, info = env.step(env.action_space)

    if terminated or truncated:
        observation, info = env.reset()


env.close()