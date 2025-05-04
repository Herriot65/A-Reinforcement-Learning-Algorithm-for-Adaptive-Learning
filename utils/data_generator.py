import json
import random
from dataclasses import asdict, dataclass
from collections import defaultdict
import math

# ====== CONFIGURATION ======
NUM_BASE_LESSONS = 5
NUM_ADVANCED_LESSONS = 25
NUM_ACTIVITIES = 100

MASTERY_MIN = 0.7
MASTERY_MAX = 0.9

ACTIVITY_CONTRIBUTION_MIN = 0.2
ACTIVITY_CONTRIBUTION_MAX = 0.5

STYLES = ['visual', 'auditory', 'kinesthetic']
DIFFICULTIES = ['easy', 'medium', 'hard']

random.seed(42)

# ====== DATA CLASSES ======
@dataclass
class Lesson:
    lesson_id: str
    required_mastery_level: float
    prerequisites: list

@dataclass
class Activity:
    activity_id: str
    max_selection_limit: int
    lesson_contributions: dict
    style: str
    difficulty: str

# ====== GENERATORS ======
def generate_lessons():
    lessons = []

    # Step 1: Base Lessons
    for i in range(NUM_BASE_LESSONS):
        lessons.append(
            Lesson(
                lesson_id=f"L{i+1}",
                required_mastery_level=round(random.uniform(MASTERY_MIN, MASTERY_MAX), 2),
                prerequisites=[]
            )
        )

    # Step 2: Advanced Lessons
    for i in range(NUM_BASE_LESSONS, NUM_BASE_LESSONS + NUM_ADVANCED_LESSONS):
        prereqs = random.sample(
            [l.lesson_id for l in lessons[:NUM_BASE_LESSONS]],
            k=random.randint(1, 2)
        )
        lessons.append(
            Lesson(
                lesson_id=f"L{i+1}",
                required_mastery_level=round(random.uniform(MASTERY_MIN, MASTERY_MAX), 2),
                prerequisites=[{p: round(random.uniform(0.7, 0.9), 2)} for p in prereqs]
            )
        )

    return lessons

def generate_activities(lessons):
    activities = []

    base_lessons = [l for l in lessons if not l.prerequisites]
    for i in range(40):
        selected = random.sample(base_lessons, k=1)
        activities.append(
            Activity(
                activity_id=f"A{i+1}",
                max_selection_limit=random.randint(2, 4),
                lesson_contributions={l.lesson_id: round(random.uniform(ACTIVITY_CONTRIBUTION_MIN, ACTIVITY_CONTRIBUTION_MAX), 2) for l in selected},
                style=random.choice(STYLES),
                difficulty="easy"
            )
        )

    for i in range(40, NUM_ACTIVITIES):
        selected = random.sample(lessons, k=random.randint(1, 2))
        difficulty = random.choice(["medium", "hard"])
        activities.append(
            Activity(
                activity_id=f"A{i+1}",
                max_selection_limit=random.randint(2, 4),
                lesson_contributions={l.lesson_id: round(random.uniform(ACTIVITY_CONTRIBUTION_MIN, ACTIVITY_CONTRIBUTION_MAX), 2) for l in selected},
                style=random.choice(STYLES),
                difficulty=difficulty
            )
        )

    return activities

def check_coherence(lessons, activities):
    total_contributions = defaultdict(float)
    for activity in activities:
        for lid, contrib in activity.lesson_contributions.items():
            total_contributions[lid] += contrib

    for lesson in lessons:
        required = lesson.required_mastery_level
        actual = total_contributions[lesson.lesson_id]
        if actual < required:
            print(f"Warning: Lesson {lesson.lesson_id} may not reach mastery ({actual:.2f} < {required:.2f})")

def fix_undercovered_lessons(lessons, activities):
    total_contributions = defaultdict(float)
    for activity in activities:
        for lid, contrib in activity.lesson_contributions.items():
            total_contributions[lid] += contrib

    for lesson in lessons:
        lid = lesson.lesson_id
        required = lesson.required_mastery_level
        provided = total_contributions[lid]
        if provided < required:
            missing = round(required - provided, 2)
            print(f" Fixing {lid} with patch activity (+{missing})")
            activities.append(
                Activity(
                    activity_id=f"A{len(activities)+1}",
                    max_selection_limit=3,
                    lesson_contributions={lid: missing},
                    style=random.choice(STYLES),
                    difficulty=random.choice(DIFFICULTIES)
                )
            )
    return activities

def generate_sprints(lessons):
    lesson_ids = [lesson.lesson_id for lesson in lessons]

    # Automatically decide number of sprints:
    # 1 sprint for every ~10 lessons
    num_sprints = max(1, len(lesson_ids) // 10)

    lessons_per_sprint = math.ceil(len(lesson_ids) / num_sprints)

    sprints = {}
    for sprint_id in range(1, num_sprints + 1):
        start_idx = (sprint_id - 1) * lessons_per_sprint
        end_idx = start_idx + lessons_per_sprint
        sprint_lessons = lesson_ids[start_idx:end_idx]
        if sprint_lessons:
            sprints[sprint_id] = {
                "lessons": sprint_lessons
            }
    return sprints

# ====== MAIN ======
def main():
    lessons = generate_lessons()
    activities = generate_activities(lessons)

    # Check and fix
    check_coherence(lessons, activities)
    activities = fix_undercovered_lessons(lessons, activities)
    check_coherence(lessons, activities)

    sprints = generate_sprints(lessons)

    # Save dataset
    dataset = {
        "lessons": [asdict(l) for l in lessons],
        "activities": [asdict(a) for a in activities],
        "sprints": sprints
    }

    with open("curriculum_adaptive_learning_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print("Curriculum generated successfully.")

if __name__ == "__main__":
    main()
