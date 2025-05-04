import sys

sys.path.append("C:/Users/herri/OneDrive/Documents/devProjects_folder/RL_for_AL")

from environments.AdaptiveLearningEnv import AdaptiveLearningEnv
from models.ppo_model.PPO import PPO
from utils.load_dataset import load_adaptive_learning_dataset
from utils.logger import TrainingLogger
from utils.config_loader import load_config

lessons, activities, sprints = load_adaptive_learning_dataset(
    filepath='data/curriculum_adaptive_learning_dataset.json'
)
env = AdaptiveLearningEnv(lessons, activities, sprints, max_activities=100)

def train_model():
    config = load_config("training/config.yaml")  
    train_cfg = config["training"]

    environment = env
    model = PPO(
        state_dim=environment.observation_space.shape[0],
        action_dim=environment.action_space.n,
        n_episodes=train_cfg["n_episodes"],
        LR=train_cfg["learning_rate"],
        GAMMA=train_cfg["gamma"],
        LAMDA=train_cfg["lamda"],
        EPOCHS=train_cfg["epochs"],
        CLIP_EPSILON=train_cfg["clip_epsilon"],
        entropy_coef=train_cfg["entropy_coef"],
        minibatch_size=train_cfg["minibatch_size"],
        freq_update=train_cfg["freq_update"]
    )

    logger = TrainingLogger(log_path=train_cfg.get("log_path", "logs/training_logs_2.csv"))

    model.train(
        environment,
        save_path=train_cfg.get("save_path", "models/checkpoints/ppo_model_20.pt"),
        logger=logger
    )

if __name__ == "__main__":
    train_model()
