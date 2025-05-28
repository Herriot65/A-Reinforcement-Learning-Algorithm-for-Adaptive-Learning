import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.train import load_curriculum

lessons, activities = load_curriculum("data/adaptive_learning_curriculum.json")

from envs.Student import Student

student = Student(dominant_style="Visual", dominant_percent=70,velocity=0.9)

action = len(activities)

#select random int in range action
import random
for _ in range(20):
    action = random.randint(0, len(activities)-1)
    print(f"Activity ID: {activities[action]['id']}  | Activity style: {activities[action]['style']}")
    print(f"Performance: {student.calculate_performance(activities[action])}")
    print("\n")

