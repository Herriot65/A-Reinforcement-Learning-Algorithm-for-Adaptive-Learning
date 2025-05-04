import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[64, 64], activation_fn=nn.ReLU):
        super(ActorCritic, self).__init__()
        
        # Build actor dynamically
        actor_layers = []
        last_dim = state_dim
        for hidden_size in hidden_sizes:
            actor_layers.append(nn.Linear(last_dim, hidden_size))
            actor_layers.append(activation_fn())
            last_dim = hidden_size
        actor_layers.append(nn.Linear(last_dim, action_dim))
        actor_layers.append(nn.Softmax(dim=-1))
        self.actor = nn.Sequential(*actor_layers)

        # Build critic dynamically
        critic_layers = []
        last_dim = state_dim
        for hidden_size in hidden_sizes:
            critic_layers.append(nn.Linear(last_dim, hidden_size))
            critic_layers.append(activation_fn())
            last_dim = hidden_size
        critic_layers.append(nn.Linear(last_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value
