"""
Main training script for solving the LunarLander-v3 environment using DQN and Double DQN.
This script initializes the environment, agents, logger, runs training, plots results, and prints a summary.

Requirements:
- gymnasium
- numpy
- matplotlib
- torch
"""
from agents.dqn_agent import DQNAgent
from utils.train_logger import TrainLogger
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

NUM_EPISODES = 1000 #number of episodes to train each agent
GAMMA = 0.99 #discount factor for return calculation

def run_training(agent, env, logger, tag="DQN"):
    """
    Runs the training loop for the specified agent
    :param agent: DQN or Double DQN agent
    :param env: environment that will be used such as Lunar Lander
    :param logger: Logger instance to record epsiode data
    :param tag: identifier tag to identify which agent is being used
    """
    #initialize for every episode
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        total_reward = 0
        episode_rewards = []
        done = False

        #run each episode
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            episode_rewards.append(reward)
            total_reward += reward

        # Calculate discounted return from episodic return
        return_g = 0
        for r in reversed(episode_rewards):
            return_g = r + GAMMA * return_g
        #Determine if lander landed successfully and give it a reward of 100
        success = reward == 100
        #logs metrics
        logger.log(episode, total_reward, return_g, success, tag)


def plot_results(logger):
    """
    Generates and saves three plots:
    - Episodic Reward vs Episode
    - Episodic Return vs Episode
    - Rolling Success Rate vs Episode
    """
    # Plot Episodic Reward
    plt.figure()
    for tag in logger.logs:
        plt.plot(logger.logs[tag]['rewards'], label=f"{tag}")
    plt.xlabel("Episode")
    plt.ylabel("Episodic Reward")
    plt.legend()
    plt.title("Episodic Reward vs Episode")
    plt.grid(True)
    plt.savefig("results/episodic_reward.png")

    # Return plot
    plt.figure()
    for tag in logger.logs:
        plt.plot(logger.logs[tag]['returns'], label=tag)
    plt.xlabel("Episode")
    plt.ylabel("Episodic Return G")
    plt.title("Episodic Return vs Episode")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/episodic_return.png")

    # Plot Success Rate
    plt.figure()
    for tag in logger.logs:
        success = np.array(logger.logs[tag]['success'])
        rolling = np.convolve(success, np.ones(10)/10, mode='valid')
        plt.plot(rolling, label=f"{tag}")
    plt.xlabel("Episode")
    plt.ylabel("Rolling Success Rate")
    plt.legend()
    plt.title("Success Rate over Time")
    plt.grid(True)
    plt.savefig("results/success_rate.png")

    plt.show()


def summarize(logger):
    """
    Prints a summary table of average reward and success rate
    over the last 100 episodes for both DQN and Double DQN.
    """
    print("\nAverage over last 100 episodes:")
    print(f"{'Metric':<30} {'DQN':>10} {'Double DQN':>15}")
    for metric in ['rewards', 'success']:
        dqn_last = np.mean(logger.logs['DQN'][metric][-100:])
        ddqn_last = np.mean(logger.logs['Double DQN'][metric][-100:])
        label = "Avg Reward" if metric == 'rewards' else "Success Rate (%)"
        dqn_val = f"{dqn_last:.2f}" if metric == 'rewards' else f"{dqn_last * 100:.1f}%"
        ddqn_val = f"{ddqn_last:.2f}" if metric == 'rewards' else f"{ddqn_last * 100:.1f}%"
        print(f"{label:<30} {dqn_val:>10} {ddqn_val:>15}")


def main():
    """
    Entry point of the script. Initializes environments, trains both DQN and Double DQN agents,
    logs metrics, generates plots, and prints a summary.
    """
    env1 = gym.make("LunarLander-v3")
    env2 = gym.make("LunarLander-v3")
    logger = TrainLogger()

    # Train Regular DQN
    dqn_agent = DQNAgent(env1, use_double_dqn=False)
    run_training(dqn_agent, env1, logger, tag="DQN")

    # Train Double DQN
    ddqn_agent = DQNAgent(env2, use_double_dqn=True)
    run_training(ddqn_agent, env2, logger, tag="Double DQN")

    #generate and save the plots
    plot_results(logger)
    #print summary table for the last 100 episodes
    summarize(logger)


if __name__ == "__main__":
    main()
