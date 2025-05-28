import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from envs.Student import Student
from envs.AdaptiveLearningEnv import AdaptiveLearningEnv

def mask_fn(env):
    return env._get_available_actions_mask()

def load_curriculum(curriculum_path):
    with open(curriculum_path) as f:
        data = json.load(f)
    return data["lessons"], data["activities"]

def train():
    # Configuration
    # curriculum_file = "data/adaptive_learning_curriculum.json"
    curriculum_file = "data/learning_curriculum.json"
    models_dir = "models/"
    logs_dir = "logs/"
    data_dir = "training_data/"
    total_timesteps = 100000
    
    # Create directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Load curriculum
    lessons, activities = load_curriculum(curriculum_file)
    
    # Initialize environments with data logging
    student = Student(dominant_style="Visual", dominant_percent=70,velocity=0.9)
    log_file = os.path.join(data_dir, "training_logs_new_dataset.json")
    
    base_env = AdaptiveLearningEnv(
        lessons, 
        activities, 
        student, 
        log_training_data=True,
        log_file=log_file
    )
    train_env = ActionMasker(base_env, mask_fn)
    
    # Train agent
    print("Starting training...")
    agent = MaskablePPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=logs_dir, 
    )
    agent.learn(total_timesteps=total_timesteps)
    
    # Finalize logging
    base_env.finalize_logging()
    
    # Save final model
    model_path = f"{models_dir}/adaptive_learning_model_50k_performance_new_dtst"
    agent.save(model_path)

if __name__ == "__main__":
    train()