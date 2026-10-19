#################################################################################
#  FrozenLake Q-learning Implementation
#  CPSC 4420 - Artificial Intelligence
#  Nadia Alexander
#################################################################################
# In this file, I reference the QLearningAgent class from the CartPole example in 
# Example3.py and adapt it for the FrozenLake environment. 
# The code includes functions to train the agent, plot the results, and test the 
# trained agent on the FrozenLake environment. 
#################################################################################
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# QLearningAgent Class (Referenced from CartPole example in Example3.py)
class QLearningAgent:
    # Initialize the Q-learning agent with default hyperparameters
    def __init__(self, n_states=64, n_actions=4, learning_rate = 0.1,
                 discount_factor= 0.99, epsilon=1.0, epsilon_decay=0.99,
                 epsilon_min=0.01):

        # Store the number of possible states and actions 
        self.n_states = n_states
        self.n_actions = n_actions

        # Store learning rate, discount factor, and exploration parameters
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize the Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))

    # Get an action based on the current state
    def get_action(self, state, training=True):
        # if training and np.random.rand() < self.epsilon, choose a random action 
        if training and np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        # Choose the action with the highest Q-value
        else:
            q_vals = self.q_table[state]
            best_actions = np.flatnonzero(q_vals == np.max(q_vals))
            return np.random.choice(best_actions)

    # Update the Q-table
    def update(self, state, action, reward, next_state, done):
        # Get the current Q-value for the state-action pair
        current_q = self.q_table[state, action]

        # If the episode is done, set the target Q-value to the reward
        if done:
            target_q = reward
        # Set the target Q-value to the reward plus the discounted maximum Q-value for the next state
        else:
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q

        # Update the Q-value for the state-action pair
        self.q_table[state, action] += self.learning_rate * (target_q - current_q)

    # Decay the epsilon after each episode
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Train the Q-learning agent on the FrozenLake environment
def train_agent(n_episodes=10000, render=False, learning_rate=0.1, discount_factor=0.99,
                epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, is_slippery=False):

    # Create the FrozenLake environment 
    env = gym.make("FrozenLake-v1", map_name = "8x8", is_slippery=is_slippery, render_mode = "human" if render else None)

    # Initialize the Q-learning agent with the specified hyperparameters
    agent = QLearningAgent(learning_rate=learning_rate, discount_factor=discount_factor,
                        epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)

    episode_rewards = []

    # Complete the learning process for all episodes
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        # Run the episode until its done, updating the Q-table and epsilon after each step
        while not done:
            # Based on the current state, get an action from the agent
            action = agent.get_action(state)
            # Take the action and save the next state, reward, and done status
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Check if the episode is done 
            done = terminated or truncated

            # Update the Q-table based on the current state, action, reward, next state, and done status
            agent.update(state, action, reward, next_state, done)

            # Update the current state and total reward for the episode
            state = next_state
            total_reward += reward

        # Decay the epsilon after each episode and save the total reward
        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        # Print early success message if the agent achieves a positive reward in the first 500 episodes
        if total_reward > 0 and episode < 500:
            print(f"Early success at episode {episode + 1} with total reward: {total_reward}")

        # Print the average reward and epsilon every 1000 episodes
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(episode_rewards[-1000:])
            print(f"Episode {episode + 1}/{n_episodes}, "
                    f"Avg Reward (last 1000): {avg_reward:.3f}, "
                    f"Epsilon: {agent.epsilon:.3f}")

        test_rewards = []
        success_steps = []

    # Evaluting 50 episodes of the trained agent for summary statistics
    for _ in range(50):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        # Run the epsiode and update Q-table and epsilon after each step
        while not done:
            action = agent.get_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            steps += 1
        test_rewards.append(total_reward)
        if total_reward > 0:
            success_steps.append(steps)

    # Close the environment and return the trained agent and episode rewards
    env.close()

    # Get the mean reward, success rate, and average steps
    mean_reward = np.mean(test_rewards)
    success_rate = mean_reward * 100.0
    avg_steps = np.mean(success_steps) if success_steps else 0.0

    print(f"Training completed and 50-epsiode evalaution completed.")
    print(f"Mean Reward: {mean_reward:.3f}, Success Rate: {success_rate:.2f}%, Average Steps: {avg_steps:.2f}")
    return agent, episode_rewards, mean_reward, success_rate, avg_steps

# Plot the results of the training process
def plot_results(episode_rewards, window = 50, title="Training Performance of the Q-learning Agent on FrozenLake"):
    plt.figure(figsize=(10, 5))

    moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')

    # Plot the moving average of the episode rewards
    plt.plot(range(window-1, len(episode_rewards)), moving_avg, label=f'{window} Episode Moving Average')

    # Add details to the plot related to the training process
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.show()

# Test the trained agent on the FrozenLake environment
def test_agent(agent, n_episodes=30):
    # Create the FrozenLake environment for testing
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False)
    test_rewards = []

    # Run the episode until its done, and update the Q-table and epsilon
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        # Run the episode until its done, and update the Q-table and epsilon
        while not done:
            action = agent.get_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward

        test_rewards.append(total_reward)
        print(f"Test Episode {episode + 1}: {total_reward:.0f} steps")

    env.close()
    avg = np.mean(test_rewards)
    print(f"\nAverage Test Reward: {avg:.1f} steps\n")

# Record of video of a trained agent's performace on the FrozenLake environment
def record_agent(agent, filename="frozenlake_q_learning_agent.npy"):
    video_folder = Path.home() / "Downloads" / "FrozenLakeVideos"
    video_folder.mkdir(parents=True, exist_ok=True)

    # RecordVideo requires an environment that returns image frames.
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=str(video_folder),name_prefix="frozenlake_q_learning_agent",)

    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    # Run the episode until its done, and update the Q-table and epsilon
    while not done:
        action = agent.get_action(state, training=False)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = next_state
        total_reward += reward
        steps += 1

    env.close()
    # Save the Q-table of the trained agent to a file
    np.save(filename, agent.q_table)
    print(f"Agent's Q-table saved to {filename}")


if __name__ == "__main__":
    #################################################################################
    # Question C: Baseline Q-Learning Agent Training
    #################################################################################

    # Use the default hyperparameters to train the Q-learning agent on the FrozenLake environment
    base_agent, base_rewards, base_mean_reward, base_success_rate, base_avg_steps = train_agent()
    plot_results(base_rewards)
    test_agent(base_agent, n_episodes=10)

    window = 50
    
    #################################################################################
    # Question D: Learning Rate Experiment
    #################################################################################
    alphas = [0.05, 0.5, 0.9]
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    for i, alpha in enumerate(alphas):
        print(f"Training with Learning Rate = {alpha}")
        agent, rewards, mean_reward, success_rate, avg_steps = train_agent(n_episodes=10000, learning_rate=alpha,
            discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
            is_slippery=False)

        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        axes[i].plot(range(window - 1, len(rewards)), moving_avg, label=f"Alpha = {alpha}")
        
        axes[i].set_xlabel("Episode")
        axes[i].set_ylabel("Average Reward")
        axes[i].set_title(f"Learning Rate Experiment (Alpha = {alpha})")
        axes[i].legend(loc="lower right")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    #################################################################################
    # Question E: Discount-Factor Experiment
    #################################################################################
    gammas = [0.50, 0.90, 0.99]
    fig_e, axes_e = plt.subplots(3, 1, figsize=(10, 12))

    for i, gamma in enumerate(gammas):
        print(f"Training with Gamma = {gamma}")
        agent, rewards, mean_reward, success_rate, avg_steps = train_agent(n_episodes=10000, discount_factor=gamma)
        
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        axes_e[i].plot(range(window - 1, len(rewards)), moving_avg, label=f"Gamma = {gamma}")
        
        axes_e[i].set_xlabel("Episode")
        axes_e[i].set_ylabel("Average Reward")
        axes_e[i].set_title(f"Discount Factor Experiment (Gamma = {gamma})")
        axes_e[i].legend(loc="lower right")
        axes_e[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    #################################################################################
    # Question F: Additional Hyperparameter Experiment (epsilon-decay)
    #################################################################################
    decays = [0.99, 0.995, 0.9995]
    fig_f, axes_f = plt.subplots(3, 1, figsize=(10, 12))

    for i, decay in enumerate(decays):
        print(f"Training with Epsilon Decay = {decay}")
        agent, rewards, mean_reward, success_rate, avg_steps = train_agent(n_episodes=10000, epsilon_decay=decay)
        
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        axes_f[i].plot(range(window - 1, len(rewards)), moving_avg, label=f"Decay = {decay}")
        
        axes_f[i].set_xlabel("Episode")
        axes_f[i].set_ylabel("Average Reward")
        axes_f[i].set_title(f"Additional Hyperparameter Experiment (Epsilon-Decay = {decay})")
        axes_f[i].legend(loc="lower right")
        axes_f[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    ################################################################################
    # Question G: Evaluating the Best Agent
    ################################################################################

    best_agent = None
    best_config = None
    max_cumulative_reward = None
    best_rewards = None

    # The combinations of hyperparameters used to answer question D, E, and F
    experiment_configs = [
        (0.05, 0.99, 0.99), (0.50, 0.99, 0.99), (0.90, 0.99, 0.99),
        (0.10, 0.50, 0.99), (0.10, 0.90, 0.99), (0.10, 0.99, 0.95),
        (0.10, 0.99, 0.99), (0.10, 0.99, 0.995), (0.10, 0.99, 0.9995)
    ]

    for alpha, gamma, decay in experiment_configs:
        print(f"Training with Alpha = {alpha}, Gamma = {gamma}, Epsilon Decay = {decay}")
        agent, rewards, mean_reward, success_rate, avg_steps = train_agent(n_episodes=10000, learning_rate=alpha,
                                                                discount_factor=gamma, epsilon_decay=decay)

        total_training_reward = sum(rewards)
        print(f" -> Cumulative Training Reward: {total_training_reward} | Success Rate: {success_rate:.1f}%")

        # Save the best agent based on the maximum cumulative reward achieved during training
        if max_cumulative_reward is None or total_training_reward > max_cumulative_reward:
            max_cumulative_reward = total_training_reward
            best_agent = agent
            best_config = (alpha, gamma, decay)
            best_rewards = rewards

    # Plot the best agent's training performance
    plot_results(best_rewards, title=f"Best Agent Training Performance (Alpha = {best_config[0]}, Gamma = {best_config[1]}, Epsilon Decay = {best_config[2]})")

    # Save video of the best agent's performance
    record_agent(best_agent, filename="best_frozenlake_q_learning_agent.npy")