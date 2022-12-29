import gym
import matplotlib.pyplot as plt

env = gym.make("BipedalWalker-v3", render_mode = 'human')

obs = env.reset()

obs_space = env.observation_space
action_space = env.action_space
# print("The observation space: {}".format(obs_space))
# print("The action space: {}".format(action_space))

print("The initial observation is {}".format(obs))

for _ in range(100):
    while True:

        random_action = env.action_space.sample()

        new_obs = env.step(random_action)
        # print("The new observation is {}".format(new_obs))

        env.render()

        # contacts = env.

        # if len(contacts > 0):
        #     env.close()
