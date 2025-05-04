from .PPO import PPO
from utils.config_loader import load_config  

def load_trained_model(env, model_path, config_path="../training/config.yaml"):
    config = load_config(config_path)
    train_cfg = config['training']

    model = PPO(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        n_episodes=train_cfg['n_episodes'],
        LR=train_cfg['learning_rate'],
        GAMMA=train_cfg['gamma'],
        LAMDA=train_cfg['lamda'],
        EPOCHS=train_cfg['epochs'],
        CLIP_EPSILON=train_cfg['clip_epsilon'],
        entropy_coef=train_cfg['entropy_coef'],
        minibatch_size=train_cfg['minibatch_size'],
        freq_update=train_cfg['freq_update']
    )
    model.load(model_path)
    return model
