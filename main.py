import os
import sys
import config 

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.Student import Student
from training.agent_trainer import train

if __name__ == "__main__":
    students_to_train = [
        Student(dominant_style="Read/Write", dominant_percent=85, velocity=0.45),
        Student(dominant_style="Read/Write", dominant_percent=85, velocity=0.75),
        Student(dominant_style="Read/Write", dominant_percent=85, velocity=0.9),
        Student(dominant_style="Visual", dominant_percent=80, velocity=0.4),
        Student(dominant_style="Visual", dominant_percent=80, velocity=0.65),
        Student(dominant_style="Visual", dominant_percent=80, velocity=0.9)
    ]

    training_results = train(
        students=students_to_train,
        total_timesteps=config.DEFAULT_TOTAL_TIMESTEPS, 
        log_file_name_json=config.LOG_FILE_NAME_JSON,
        model_name=config.MODEL_NAME,
        test_after_training=config.TEST_AFTER_TRAINING,
        analyze_progress=config.ANALYZE_PROGRESS,
        test_episodes=config.TEST_EPISODES,
    )

    print("\n--- Training Summary ---")
    for result in training_results:
        print(f"Student ID: {result['student_id']}")
        print(f"  Model Path: {result['model_path']}")
        print(f"  Log File: {result['log_file']}")
        print(f"  Output Directory: {result['output_dir']}")
        print("-" * 30)