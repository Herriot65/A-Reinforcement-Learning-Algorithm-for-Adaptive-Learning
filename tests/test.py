import csv
import os
import numpy as np
import sys
sys.path.append("C:/Users/herri/OneDrive/Documents/devProjects_folder/RL_for_AL")
from environments.AdaptiveLearningEnv import AdaptiveLearningEnv
from utils.data_generator import STYLES
from utils.load_dataset import load_adaptive_learning_dataset
from models.ppo_model.utils import load_trained_model

lessons, activities, sprints = load_adaptive_learning_dataset(filepath="../data/curriculum_adaptive_learning_dataset.json")

def generate_student_profile():
    velocity = round(np.clip(np.random.normal(0.6, 0.1), 0.4, 0.9), 2)
    style = np.random.choice(STYLES)
    return style, velocity

def create_student_env(style, velocity, max_activities):
    env = AdaptiveLearningEnv(lessons, activities, sprints, max_activities=max_activities)
    env.velocity = velocity
    env.style = style
    env.style_one_hot = env._encode_style(style)
    return env

def test_model(model_path, num_students=5, max_activities=100, output_file="logs/test_results.csv"):
    print("\n=== TESTING TRAINED PPO ON ARTIFICIAL STUDENTS ===\n")

    # Prepare CSV writing
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fieldnames = ["student_id", "style", "velocity", "steps", "total_reward"]
    lesson_ids = sorted(lessons.keys())
    fieldnames += [f"mastery_{lid}" for lid in lesson_ids]

    with open(output_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        sample_env = create_student_env("visual", 0.6, max_activities)
        model = load_trained_model(sample_env, model_path)

        for student_id in range(1, num_students + 1):
            style, velocity = generate_student_profile()
            env = create_student_env(style, velocity, max_activities)

            obs, _ = env.reset()
            done, total_reward, steps = False, 0, 0

            while not done:
                action, _ = model.select_action(obs, deterministic=True)
                obs, reward, done, _, _ = env.step(action)
                total_reward += reward
                steps += 1

            # Collect result row
            result = {
                "student_id": student_id,
                "style": style,
                "velocity": velocity,
                "steps": steps,
                "total_reward": round(total_reward, 2)
            }
            for lid in lesson_ids:
                result[f"mastery_{lid}"] = round(env.mastery.get(lid, 0.0), 3)

            writer.writerow(result)

            # Also print to console
            print(f"Student {student_id} | Style: {style} | Velocity: {velocity}")
            print(f"Finished in {steps} steps | Total Reward: {round(total_reward, 2)}")
            print("Final Mastery Levels:")
            for lid in lesson_ids:
                mastery = env.mastery.get(lid, 0.0)
                status = "✅" if mastery >= lessons[lid]["required_mastery_level"] else "❌"
                print(f" - {lid}: {mastery:.2f} / {lessons[lid]['required_mastery_level']} {status}")
            print()

if __name__ == "__main__":
    test_model(
        model_path="models/checkpoints/ppo_model.pt",
        num_students=5,
        max_activities=100,
        output_file="logs/test_results.csv"
    )
