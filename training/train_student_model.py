import os
import sys
import json
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.Student import Student
from envs.AdaptiveLearningEnv import AdaptiveLearningEnv
from tests.learning_progress_analyzer import analyze_learning_progress
from tests.student_model_tester import ModelTester

# Ignore warnings 
import torch.distributions
torch.distributions.Distribution._validate_args = False

def mask_fn(env):
    return env._get_available_actions_mask()

def load_curriculum(curriculum_path):
    with open(curriculum_path) as f:
        data = json.load(f)
    return data["lessons"], data["activities"]

def make_env(lessons, activities, log_file, student, max_steps=1000): 
    def _init():
        env = AdaptiveLearningEnv(
            lessons,
            activities,
            student=student,
            log_training_data=True,
            log_file=log_file,
            max_steps=max_steps
        )
        return ActionMasker(env, mask_fn)
    return _init

def train_single_student(
    curriculum_file,
    models_dir,
    logs_dir,
    data_dir,
    total_timesteps,
    n_envs,
    log_file_name_json,
    model_name,
    student,
    learning_rate=1e-4,
    batch_size=64,
    n_epochs=10,
    clip_range=0.2,
    n_steps=2048,
    gamma=0.95,
    gae_lambda=0.97,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    test_after_training=True,
    test_episodes=1,
    analyze_progress=True,
    max_steps = 1000
):
    """Train a single student model with PPO."""
   
    lessons, activities = load_curriculum(curriculum_file)
    
    student_id = f"{student.dominant_style}_{student.dominant_percent}pct_{str(student.velocity).replace('.', '')}vel".replace("/", "_")

    log_file = os.path.join(data_dir, f"{log_file_name_json.replace('.json', '')}_{student_id}.json")
    model_path = os.path.join(models_dir, f"{model_name}_{student_id}")
    unified_output_dir = f"tests/model_test_results/{model_name}_{student_id}/"
    os.makedirs(unified_output_dir, exist_ok=True)

    # Create vectorized environment
    train_env = make_vec_env(
        make_env(lessons, activities, log_file, student=student, max_steps=max_steps), 
        n_envs=n_envs
    )

    # Initialize and train PPO agent
    print(f"Starting training for student: {student_id}")
    agent = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=verbose,
        tensorboard_log=f"{logs_dir}/{student_id}",
    )

    # agent.learn(total_timesteps=total_timesteps)
    # agent.save(model_path)
    # print(f"Model saved to: {model_path}")

    # Test the trained model
    if test_after_training:
        print(f"Testing trained model for {student_id}...")
        try:
            tester = ModelTester(
                model_path=model_path,
                curriculum_path=curriculum_file,
                output_dir=unified_output_dir,
                mastery_threshold=1.0
            )
            _ = tester.test_student(student, num_episodes=test_episodes, student_name=f"{student_id}_trained")

        except Exception as e:
            print(f"Error during model testing: {e}")

    if analyze_progress:
        print(f"Analyzing learning progress for {student_id}...")
        try:
            analyze_learning_progress(log_file, unified_output_dir)
        except Exception as e:
            print(f"Error during learning progress analysis: {e}")

    metadata = {
        "algorithm": "PPO",
        "total_timesteps_trained": total_timesteps,
        "model_name": model_name,
        "n_envs": n_envs,
        "max_steps": max_steps,
        "curriculum_file": curriculum_file,
        "hyperparameters": {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm
        },
        "student_config": {
            "dominant_style": student.dominant_style,
            "dominant_percent": student.dominant_percent,
            "velocity": student.velocity
        }
    }

    with open(f"{unified_output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "log_file": log_file,
        "model_path": model_path,
        "output_dir": unified_output_dir,
        "student_id": student_id
    }

def train(
    curriculum_file="data/learning_curriculum.json",
    models_dir="models/",
    logs_dir="logs/",
    data_dir="training_data/",
    total_timesteps=100000,
    n_envs=4,
    log_file_name_json="student_data.json",
    model_name="student_model",
    students=None,
    learning_rate=1e-4,
    batch_size=64,
    n_epochs=10,
    clip_range=0.2,
    n_steps=2048,
    gamma=0.95,
    gae_lambda=0.97,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    test_after_training=True,
    test_episodes=1,
    analyze_progress=True,
):
    """
    Train PPO agents for multiple students.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    if students is None:
        students = [Student(dominant_style="Visual", dominant_percent=80, velocity=0.65)]
    elif not isinstance(students, list):
        students = [students]

    results = []
    for student in students:
        if student.velocity <= 0.45:
            total_timesteps = 250000
        result = train_single_student(
            curriculum_file=curriculum_file,
            models_dir=models_dir,
            logs_dir=logs_dir,
            data_dir=data_dir,
            total_timesteps=total_timesteps,
            n_envs=n_envs,
            log_file_name_json=log_file_name_json,
            model_name=model_name,
            student=student,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_epochs=n_epochs,
            clip_range=clip_range,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
            test_after_training=test_after_training,
            test_episodes=test_episodes,
            analyze_progress=analyze_progress
        )
        results.append(result)

    return results

if __name__ == "__main__":
    students = [
        Student(dominant_style="Read/Write", dominant_percent=85, velocity=0.45),
        Student(dominant_style="Read/Write", dominant_percent=85, velocity=0.75),
        Student(dominant_style="Read/Write", dominant_percent=85, velocity=0.9),
        Student(dominant_style="Visual", dominant_percent=80, velocity=0.4),
        Student(dominant_style="Visual", dominant_percent=80, velocity=0.65),
        Student(dominant_style="Visual", dominant_percent=80, velocity=0.9)
    ]

    results = train(
        total_timesteps=200000,
        log_file_name_json="student_data.json",
        model_name="student_model",
        students=students,
        test_after_training=True,
        analyze_progress=True,
        test_episodes=1,
    )
    