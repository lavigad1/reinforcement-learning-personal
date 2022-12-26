import random

# define the grid environment
# 0 = empty space, 1 = wall, 2 = goal
grid = [[[0, 0, 0, 1],
         [0, 1, 0, 1],
         [0, 1, 0, 0],
         [0, 0, 2, 0]],
        [[0, 0, 0, 0],
         [0, 1, 0, 1],
         [0, 1, 0, 0],
         [0, 0, 2, 0]],
        [[0, 0, 0, 0],
         [0, 1, 0, 1],
         [0, 1, 0, 0],
         [0, 0, 2, 0]]]

# define the action space
actions = ["up", "down", "left", "right", "forward", "backward"]

# define the agent's starting position
x = 0
y = 0
z = 0

# define the maximum number of steps the agent can take
max_steps = 10

# define the reward for reaching the goal
goal_reward = 10

# define the penalty for hitting a wall or taking too many steps
collision_penalty = -1

# define the learning rate
alpha = 0.1

# define the discount factor
gamma = 0.9

# initialize the Q-table to all zeros
q_table = [[[[0 for action in actions] for col in row] for row in grid_slice] for grid_slice in grid]

# define the number of episodes to train the agent
num_episodes = 1000

# define the exploration rate
exploration_rate = 0.1

# define the exploration decay rate
exploration_decay_rate = 0.999

for episode in range(num_episodes):
  # reset the environment and the agent's position
  x = 0
  y = 0
  z = 0
  grid[x][y][z] = 0
  step = 0
  total_reward = 0
  
  while True:
    # choose an action using the exploration-exploitation trade-off
    if random.uniform(0, 1) < exploration_rate:
      # explore by choosing a random action
      action = random.choice(actions)
    else:
      # exploit by choosing the action with the highest Q-value
      action = actions[q_table[x][y][z].index(max(q_table[x][y][z]))]
      
    # update the agent's position based on the chosen action
    if action == "up":
      x -= 1
    elif action == "down":
      x += 1
    elif action == "left":
      y -= 1
    elif action == "right":
      y += 1
    elif action == "forward":
      z += 1
    elif action == "backward":
      z -= 1
      
    # check if the agent has hit a wall or reached the goal
    if grid[x][y][z] == 0:
      # the agent has hit a wall
      reward = collision_penalty
    elif grid[x][y][z] == 2:
      # the agent has reached the goal
      reward = goal_reward
    else:
      # the agent has neither hit a wall nor reached the goal
      reward = 0
      
    # update the Q-table
    q_table[x][y][z][actions.index(action)] += alpha * (reward + gamma * max(q_table[x][y][z]) - q_table[x][y][z][actions.index(action)])
      
    # update the total reward
    total_reward += reward
      
    # check if the agent has hit a wall or reached the goal
    if grid[x][y][z] == 1 or grid[x][y][z] == 2:
      # the agent has either hit a wall or reached the goal
      break
      
    # update the number of steps taken
    step += 1
    
    # check if the maximum number of steps has been exceeded
    if step >= max_steps:
      # the maximum number of steps has been exceeded
      break
      
  # update the exploration rate
  exploration_rate *= exploration_decay_rate
  
  # print the episode summary
  print("Episode #: {0} \t Reward: {1} \t Exploration Rate: {2:.2f}".format(episode + 1, total_reward, exploration_rate))
  
# print the final Q-table
print("\nFinal Q-table:")
for grid_slice in q_table:
  print(grid_slice)
