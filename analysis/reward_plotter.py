import pandas as pd
import matplotlib.pyplot as plt

def plot_rewards_from_csv(csv_path):
    """
    Plots episode reward evolution from a CSV log file.

    Args:
        csv_path (str): Path to the CSV file containing episode-level logs.
    """
    df = pd.read_csv(csv_path)

    # Group by episode and get the final reward of each episode
    rewards_by_episode = df.groupby('episode')['total_episode_reward'].max().reset_index()

    plt.figure(figsize=(10, 5))
    plt.plot(rewards_by_episode['episode'], rewards_by_episode['total_episode_reward'], marker='o')
    plt.title("Episode Reward Evolution")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
