# utils/train_logger.py
class TrainLogger:
    """
    logger to track and store training metrics per episode.
    """
    def __init__(self):
        """
        initialize the logger with an array
        """
        self.logs = {}  # Dict[str, Dict[str, List]]

    def log(self, episode, reward, return_g, success, tag):
        """
        logs the metrics for a single episode
        :param episode: current episode number
        :param reward: total episode rewards
        :param return_g: discounted return G value
        :param success: boolean whether the episode ended successfully or not
        :param tag: identifier for the agent DQN or Double DQN
        """
        #initialize log storage for a new agent tag
        if tag not in self.logs:
            self.logs[tag] = {'rewards': [], 'returns': [], 'success': []}
        self.logs[tag]['rewards'].append(reward)
        self.logs[tag]['returns'].append(return_g)
        self.logs[tag]['success'].append(success)
        #prints every 10 episodes the required information
        if episode % 10 == 0:
            print(f"[Episode {episode}] {tag} - Reward: {reward:.1f} | Return G: {return_g:.1f}")


