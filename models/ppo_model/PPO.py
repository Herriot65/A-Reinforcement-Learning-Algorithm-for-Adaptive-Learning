import torch
import os 
import torch.optim as optim
import numpy as np
import random
from .ActorCritic import ActorCritic

class PPO:
    def __init__(self, state_dim, action_dim,n_episodes, LR, GAMMA, LAMDA, EPOCHS, CLIP_EPSILON,
                 entropy_coef=0.01, minibatch_size=64, freq_update=2048):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        
        self.memory = []
        self.GAMMA = GAMMA
        self.LAMDA = LAMDA
        self.EPOCHS = EPOCHS
        self.CLIP_EPSILON = CLIP_EPSILON
        self.entropy_coef = entropy_coef
        self.minibatch_size = minibatch_size
        self.freq_update = freq_update
        self.n_episodes = n_episodes

    def select_action(self, state, deterministic=False):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.policy(state)
        dist = torch.distributions.Categorical(action_probs)
        action = torch.argmax(action_probs).item() if deterministic else dist.sample().item()
        return action, action_probs[action].item()

    def store(self, transition):
        self.memory.append(transition)

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.GAMMA * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.GAMMA * self.LAMDA * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self):
        states, actions, old_probs, rewards, dones = zip(*self.memory)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        old_probs = torch.tensor(old_probs, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            _, values = self.policy(states)
            values = values.squeeze()

        values_for_adv = values.tolist() + [0.0]
        advantages = self.compute_advantages(rewards, values_for_adv, dones)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = advantages + values.detach()

        for _ in range(self.EPOCHS):
            indices = list(range(len(states)))
            random.shuffle(indices)
            for start in range(0, len(states), self.minibatch_size):
                end = start + self.minibatch_size
                minibatch_idx = indices[start:end]

                batch_states = states[minibatch_idx]
                batch_actions = actions[minibatch_idx]
                batch_old_probs = old_probs[minibatch_idx]
                batch_returns = returns[minibatch_idx]
                batch_advantages = advantages[minibatch_idx]

                action_probs, new_values = self.policy(batch_states)
                new_values = new_values.squeeze()
                new_probs = action_probs.gather(1, batch_actions).squeeze()

                ratio = new_probs / batch_old_probs
                clipped_ratio = torch.clamp(ratio, 1 - self.CLIP_EPSILON, 1 + self.CLIP_EPSILON)
                policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()

                value_loss = (batch_returns - new_values).pow(2).mean()
                entropy = torch.distributions.Categorical(action_probs).entropy().mean()

                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.memory = []

    def save(self, path):
        torch.save({
            'actor': self.policy.actor.state_dict(),
            'critic': self.policy.critic.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.actor.load_state_dict(checkpoint['actor'])
        self.policy.critic.load_state_dict(checkpoint['critic'])
        return self

    def train(self, env, save_path=None, logger=None):
        episode_rewards = []
        total_timesteps = 0

        for episode in range(self.n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            step = 0

            while not done:
                action, action_prob = self.select_action(obs)
                next_obs, reward, done, _, info = env.step(action)

                # Store transition for PPO update
                self.store((obs, action, action_prob, reward, done))
                episode_reward += reward
                
                activity_id = list(env.activities.keys())[action]
                affected_lessons = env.activities[activity_id]["lesson_contributions"]
                mastery_gains = {}
                for lid, mastery_gain in env.mastery.items():
                    if mastery_gain > 0:
                        mastery_gains[lid] = mastery_gain

                # Log the step if logger is provided
                if logger:
                    logger.log_step(
                        episode=episode,
                        step=step,
                        action_id= activity_id,
                        affected_lessons=affected_lessons,
                        mastery_gains=mastery_gains,
                        reward=reward,
                        episode_reward=episode_reward,
                        done=done,
                        info=info
                    )

                obs = next_obs
                step += 1
                total_timesteps += 1

                if total_timesteps % self.freq_update == 0:
                    self.update()

            episode_rewards.append(episode_reward)

            if logger:
                logger.flush()

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1} | Reward: {round(episode_reward, 2)}")

        if self.memory:
            self.update()

        if save_path:
            self.save(save_path)
            print(f"Model saved to {save_path}")

        return episode_rewards

