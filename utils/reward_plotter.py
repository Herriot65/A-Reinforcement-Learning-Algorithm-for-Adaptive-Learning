import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import make_interp_spline

def plot_cumulative_reward(json_file_path, output_dir, filter_incomplete=True, min_steps_threshold=10):
    """
    Create a smooth, appealing plot of cumulative reward per episode showing training evolution.
    
    Args:
        json_file_path: Path to the training data JSON file
        output_dir: Directory to save the plot
        filter_incomplete: Whether to filter out incomplete episodes (default: True)
        min_steps_threshold: Minimum number of steps to consider an episode complete (default: 10)
    """
    
    # Load JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Calculate cumulative reward per episode
    episodes = []
    cumulative_rewards = []
    episode_lengths = []
    
    for episode in data['episodes']:
        episode_num = episode['episode']
        episode_steps = episode['steps']
        episode_rewards = [step['performance'] for step in episode_steps]
        total_reward = sum(episode_rewards)
        
        episodes.append(episode_num)
        cumulative_rewards.append(total_reward)
        episode_lengths.append(len(episode_steps))
    
    # Convert to numpy arrays
    episodes = np.array(episodes)
    cumulative_rewards = np.array(cumulative_rewards)
    episode_lengths = np.array(episode_lengths)
    
    # Filter out incomplete episodes if requested
    if filter_incomplete and len(episodes) > 1:
        # Method 1: Remove episodes that are significantly shorter than the median
        median_length = np.median(episode_lengths)
        length_threshold = max(min_steps_threshold, median_length * 0.3)  # 30% of median or min threshold
        
        # Method 2: Also check if the last few episodes are unusually short
        if len(episodes) >= 5:
            # Check if last episode is much shorter than recent average
            recent_avg_length = np.mean(episode_lengths[-10:-1])  # Average of episodes -10 to -2
            last_episode_length = episode_lengths[-1]
            
            # If last episode is less than 50% of recent average, it's likely incomplete
            if last_episode_length < recent_avg_length * 0.5:
                # Remove the last episode
                episodes = episodes[:-1]
                cumulative_rewards = cumulative_rewards[:-1]
                episode_lengths = episode_lengths[:-1]
        
        # Apply general length filter
        valid_mask = episode_lengths >= length_threshold
        episodes = episodes[valid_mask]
        cumulative_rewards = cumulative_rewards[valid_mask]
        episode_lengths = episode_lengths[valid_mask]
        
        filtered_count = len(valid_mask) - np.sum(valid_mask)
    
    # Create the plot with modern styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create smooth interpolation for the main curve
    if len(episodes) > 3:  # Need at least 4 points for spline
        # Create more points for smoother curve
        episodes_smooth = np.linspace(episodes.min(), episodes.max(), 300)
        spl = make_interp_spline(episodes, cumulative_rewards, k=3)
        rewards_smooth = spl(episodes_smooth)
        
        # Plot smooth curve
        ax.plot(episodes_smooth, rewards_smooth, 
                linewidth=3, color='#2E86C1', alpha=0.8, 
                label='Training Progress')
    else:
        # Fallback for few data points
        ax.plot(episodes, cumulative_rewards, 
                linewidth=3, color='#2E86C1', alpha=0.8,
                label='Training Progress')
    
    # Add actual data points
    ax.scatter(episodes, cumulative_rewards, 
               s=60, color='#1B4F72', alpha=0.7, 
               edgecolors='white', linewidth=2, 
               zorder=5, label='Episodes')
    
    # Add trend line
    z = np.polyfit(episodes, cumulative_rewards, 1)
    trend_line = np.poly1d(z)
    ax.plot(episodes, trend_line(episodes), 
            '--', color='#E74C3C', alpha=0.8, linewidth=2.5,
            label=f'Trend (slope: {z[0]:.2f})')
    
    # Fill area under curve for visual appeal
    if len(episodes) > 3:
        ax.fill_between(episodes_smooth, rewards_smooth, 
                       alpha=0.2, color='#2E86C1')
    else:
        ax.fill_between(episodes, cumulative_rewards, 
                       alpha=0.2, color='#2E86C1')
    
    # Enhanced styling
    title = 'Training Evolution: Cumulative Reward per Episode'
    # if filter_incomplete:
    #     title += ' (Incomplete Episodes Filtered)'
    
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Episode Number', fontsize=14, fontweight='semibold')
    ax.set_ylabel('Cumulative Reward', fontsize=14, fontweight='semibold')
    
    # Improve grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_facecolor('#FAFAFA')
    
    # Add legend
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=11)
    
    # Set axis limits with some padding
    y_range = max(cumulative_rewards) - min(cumulative_rewards)
    ax.set_ylim(min(cumulative_rewards) - 0.05 * y_range, 
                max(cumulative_rewards) + 0.05 * y_range)
    
    # Add subtle annotation for best performance
    best_idx = np.argmax(cumulative_rewards)
    best_episode = episodes[best_idx]
    best_reward = cumulative_rewards[best_idx]
    
    ax.annotate(f'Peak: Episode {best_episode}\nReward: {best_reward:.1f}',
                xy=(best_episode, best_reward),
                xytext=(best_episode + (max(episodes) - min(episodes)) * 0.15, 
                       best_reward + y_range * 0.05),
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor='#E74C3C', alpha=0.8),
                fontsize=10, ha='left')
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot with high quality
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plot_file = output_path / 'cumulative_reward_evolution.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    # plt.show()
    
    return str(plot_file)

def plot_cumulative_reward_with_ma(json_file_path, output_dir, window_size=5, filter_incomplete=True):
    """
    Create plot with moving average smoothing and incomplete episode filtering.
    """
    
    # Load and process data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    episodes = []
    cumulative_rewards = []
    episode_lengths = []
    
    for episode in data['episodes']:
        episode_num = episode['episode']
        episode_steps = episode['steps']
        episode_rewards = [step['performance'] for step in episode_steps]
        total_reward = sum(episode_rewards)
        
        episodes.append(episode_num)
        cumulative_rewards.append(total_reward)
        episode_lengths.append(len(episode_steps))
    
    # Filter incomplete episodes
    if filter_incomplete and len(episodes) > 1:
        episode_lengths = np.array(episode_lengths)
        
        # Remove last episode if it's significantly shorter
        if len(episodes) >= 5:
            recent_avg_length = np.mean(episode_lengths[-10:-1])
            last_episode_length = episode_lengths[-1]
            
            if last_episode_length < recent_avg_length * 0.5:
                episodes = episodes[:-1]
                cumulative_rewards = cumulative_rewards[:-1]
    
    # Convert to pandas for easy moving average
    df = pd.DataFrame({'episode': episodes, 'reward': cumulative_rewards})
    df['ma_reward'] = df['reward'].rolling(window=window_size, center=True).mean()
    
    # Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Raw data (more transparent)
    ax.plot(df['episode'], df['reward'], 
            alpha=0.4, color='#85C1E9', linewidth=1, 
            label='Raw Data')
    
    # Moving average (prominent)
    ax.plot(df['episode'], df['ma_reward'], 
            linewidth=4, color='#2E86C1', 
            label=f'Moving Average (window={window_size})')
    
    # Scatter points
    ax.scatter(df['episode'], df['reward'], 
               s=40, alpha=0.6, color='#1B4F72', 
               edgecolors='white', linewidth=1)
    
    # Styling
    title = f'Training Evolution with {window_size}-Episode Moving Average'
    if filter_incomplete:
        title += ' (Filtered)'
        
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Episode Number', fontsize=14)
    ax.set_ylabel('Cumulative Reward', fontsize=14)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_file = output_path / f'ma_cumulative_rewards_w{window_size}_filtered.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    # plt.show()
    
    return str(plot_file)

# Alternative: Simple last-N episodes removal
def plot_cumulative_reward_simple_filter(json_file_path, output_dir, remove_last_n=1):
    """
    Simple approach: just remove the last N episodes to avoid incomplete episode issues.
    """
    
    # Load JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Calculate cumulative reward per episode
    episodes = []
    cumulative_rewards = []
    
    for episode in data['episodes']:
        episode_num = episode['episode']
        episode_rewards = [step['performance'] for step in episode['steps']]
        total_reward = sum(episode_rewards)
        
        episodes.append(episode_num)
        cumulative_rewards.append(total_reward)
    
    # Simple filter: remove last N episodes
    if remove_last_n > 0 and len(episodes) > remove_last_n:
        episodes = episodes[:-remove_last_n]
        cumulative_rewards = cumulative_rewards[:-remove_last_n]
    
    # Convert to numpy arrays
    episodes = np.array(episodes)
    cumulative_rewards = np.array(cumulative_rewards)
    
    # Rest of the plotting code remains the same as the original function
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create smooth interpolation
    if len(episodes) > 3:
        episodes_smooth = np.linspace(episodes.min(), episodes.max(), 300)
        spl = make_interp_spline(episodes, cumulative_rewards, k=3)
        rewards_smooth = spl(episodes_smooth)
        ax.plot(episodes_smooth, rewards_smooth, 
                linewidth=3, color='#2E86C1', alpha=0.8, 
                label='Training Progress')
        ax.fill_between(episodes_smooth, rewards_smooth, 
                       alpha=0.2, color='#2E86C1')
    else:
        ax.plot(episodes, cumulative_rewards, 
                linewidth=3, color='#2E86C1', alpha=0.8,
                label='Training Progress')
    
    # Add points and trend
    ax.scatter(episodes, cumulative_rewards, s=60, color='#1B4F72', alpha=0.7, 
               edgecolors='white', linewidth=2, zorder=5, label='Episodes')
    
    z = np.polyfit(episodes, cumulative_rewards, 1)
    trend_line = np.poly1d(z)
    ax.plot(episodes, trend_line(episodes), '--', color='#E74C3C', 
            alpha=0.8, linewidth=2.5, label=f'Trend (slope: {z[0]:.2f})')
    
    # Styling
    ax.set_title(f'Training Evolution (Last {remove_last_n} Episode(s) Removed)', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Episode Number', fontsize=14, fontweight='semibold')
    ax.set_ylabel('Cumulative Reward', fontsize=14, fontweight='semibold')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_facecolor('#FAFAFA')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=11)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_file = output_path / f'cumulative_rewards_remove_last_{remove_last_n}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    return str(plot_file)

if __name__ == "__main__":
    # Example usage with different filtering approaches
    json_path = "training_data/student_data_visual_80pct_vel065.json"
    output_path = "test_results/student_model_visual_80pct_vel065/"
    
    # Method 1: Smart filtering based on episode length
    plot_cumulative_reward(json_path, output_path, filter_incomplete=True)
    
    # Method 2: Simple removal of last episode
    # plot_cumulative_reward_simple_filter(json_path, output_path, remove_last_n=1)
    
    # Method 3: Moving average with filtering
    # plot_cumulative_reward_with_ma(json_path, output_path, window_size=5, filter_incomplete=True)