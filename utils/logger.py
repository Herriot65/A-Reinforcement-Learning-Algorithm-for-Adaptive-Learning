import csv
import os

class TrainingLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        self.fields = [
            "episode", "step", "action_id", "affected_lessons",
            "mastery_gains", "reward", "total_episode_reward", "done","info"
        ]
        with open(self.log_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()

        self.buffer = []

    def log_step(self, episode, step, action_id, affected_lessons,
                 mastery_gains, reward, episode_reward, done, info):
        self.buffer.append({
            "episode": episode,
            "step": step,
            "action_id": action_id,
            "affected_lessons": ";".join(affected_lessons),
            "mastery_gains": mastery_gains,
            "reward": round(reward, 3),
            "total_episode_reward": round(episode_reward, 3),
            "done": done,
            "info": info
        })

    def flush(self):
        if not self.buffer:
            return
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerows(self.buffer)
        self.buffer = []
