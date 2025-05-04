import sys
sys.path.append("C:/Users/herri/OneDrive/Documents/devProjects_folder/RL_for_AL")
from analysis.reward_plotter import plot_rewards_from_csv

plot_rewards_from_csv("../logs/training_logs_30000.csv")
