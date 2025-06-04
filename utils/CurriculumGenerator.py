import os
import sys
import json
import random
from typing import Dict, List
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CurriculumGenerator:
    """
    Creates a synthetic curriculum JSON for the adaptive-learning RL sandbox.

    Each activity now includes:
        • lessons   : {lesson_id: coverage}
        • style     : {V, A, R, K} (sums to 1)
        • nb_points : int  (points awarded if fully correct)
    """

    def __init__(
        self,
        num_lessons: int = 4,
        activities_per_lesson: int = 200,
        learning_styles: List[str] = ("visual", "auditory", "read_write", "kinesthetic"),
        max_prerequisites: int = 2,
        max_lessons_per_activity: int = 3,
        points: int = 10,   
    ):
        self.num_lessons = num_lessons
        self.activities_per_lesson = activities_per_lesson
        self.learning_styles = list(learning_styles)
        self.max_prerequisites = max_prerequisites
        self.max_lessons_per_activity = max_lessons_per_activity
        self.points = points

    # ───────────────────────────────────────────────────────── helpers ──
    @staticmethod
    def generate_lesson_id(idx: int) -> str:
        return f"L{idx+1}"

    @staticmethod
    def generate_activity_id(idx: int) -> str:
        return f"A{idx+1}"

    def generate_learning_style(self) -> Dict[str, float]:
        """Random VARK vector (2-decimals, sums to 1)."""
        while True:
            vec = {s: round(random.random(), 2) for s in self.learning_styles}
            total = sum(vec.values())
            if total > 0:
                return {k: round(v / total, 2) for k, v in vec.items()}

    # ───────────────────────────────────────────────────── lessons ──
    def generate_lessons(self) -> List[Dict]:
        lessons = []
        for i in range(self.num_lessons):
            lesson_id = self.generate_lesson_id(i)
            prerequisites = {}

            if i:                                           # L1 has none
                num = min(i, random.randint(1, self.max_prerequisites))
                for prereq in random.sample(
                    [self.generate_lesson_id(j) for j in range(i)], num
                ):
                    prerequisites[prereq] = round(random.uniform(0.4, 0.8), 2)

            lessons.append({"id": lesson_id, "prerequisites": prerequisites})
        return lessons

    # ───────────────────────────────────────────────── activity helpers ──
    def generate_activity_lessons(self, focal: str) -> Dict[str, float]:
        """Coverage dict that always includes `focal`."""
        if random.randint(1, self.max_lessons_per_activity) == 1:
            return {focal: 1.0}

        main_cov = round(random.uniform(0.5, 0.9), 2)
        others = [self.generate_lesson_id(i) for i in range(self.num_lessons) if self.generate_lesson_id(i) != focal]
        n_others = random.randint(1, self.max_lessons_per_activity - 1)
        selected = random.sample(others, n_others)

        while True:
            weights = [round(random.random(), 2) for _ in selected]
            tot = sum(weights)
            if tot:
                coverage = {focal: main_cov}
                rem = round(1.0 - main_cov, 2)
                for w, les in zip(weights, selected):
                    coverage[les] = round(rem * w / tot, 2)
                # re-normalize
                s = sum(coverage.values())
                if all(0 < v <= 1 for v in coverage.values()):
                    return {k: round(v / s, 2) for k, v in coverage.items()}

    # ────────────────────────────────────────────────── activities ──
    def generate_activities(self, lessons: List[Dict]) -> List[Dict]:
        acts: List[Dict] = []
        aid = 0

        # Focused activities per lesson
        for lesson in lessons:
            for _ in range(self.activities_per_lesson):
                acts.append(
                    {
                        "id": self.generate_activity_id(aid),
                        "lessons": self.generate_activity_lessons(lesson["id"]),
                        "style": self.generate_learning_style(),
                        "nb_points": self.points  # NEW
                    }
                )
                aid += 1

        # 10 % mixed review
        for _ in range(int(len(acts) * 0.1)):
            select = random.sample([l["id"] for l in lessons],
                                   random.randint(2, self.max_lessons_per_activity))
            while True:
                w = [round(random.random(), 2) for _ in select]
                if (tot := sum(w)):
                    cov = {l: round(v / tot, 2) for l, v in zip(select, w)}
                    if abs(sum(cov.values()) - 1.0) < 1e-3:
                        acts.append(
                            {
                                "id": self.generate_activity_id(aid),
                                "lessons": cov,
                                "style": self.generate_learning_style(),
                                "nb_points": self.points,
                            }
                        )
                        aid += 1
                        break
        return acts

    # ───────────────────────────────────────────── curriculum ──
    def generate_curriculum(self, file_path: str | None = None) -> Dict:
        lessons = self.generate_lessons()
        activities = self.generate_activities(lessons)
        curriculum = {
            "lessons": lessons,
            "activities": activities,
            "metadata": {
                "num_lessons": self.num_lessons,
                "total_activities": len(activities),
                "learning_styles": self.learning_styles,
                "description": "Synthetic curriculum with VARK styles and nb_points.",
            },
        }
        if file_path:
            with open(file_path, "w", encoding="utf-8") as fp:
                json.dump(curriculum, fp, indent=2)
        return curriculum


# ───────────────────────────────────────────── demo ──
if __name__ == "__main__":
    gen = CurriculumGenerator(num_lessons=4, activities_per_lesson=400)
    gen.generate_curriculum("data/balanced_curriculum.json")
    # print("Saved → data/adaptive_learning_curriculum.json")

