import os
import sys
import json
import random
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CurriculumGenerator:
    """A class for generating a synthetic learning curriculum."""
    def __init__(
        self,
        num_lessons: int = 4,
        activities_per_lesson: int = 200,
        learning_styles: List[str] = ("visual", "auditory", "read_write", "kinesthetic"),
        max_prerequisites: int = 2,
        max_lessons_per_activity: int = 3,
        points: int = 10,
    ):
        """
        Initializes the CurriculumGenerator with configuration parameters.

        Args:
            num_lessons (int): The total number of distinct lessons to be generated in the curriculum.
            activities_per_lesson (int): The base number of activities primarily focused on each lesson.
            learning_styles (List[str]): A tuple of strings defining the distinct learning styles
                                         that activities can cater to.
            max_prerequisites (int): The maximum number of prerequisite lessons any given lesson can have.
            max_lessons_per_activity (int): The maximum number of distinct lessons a single activity can cover.
            points (int): The fixed number of points awarded for completing any activity.
        """
        self.num_lessons = num_lessons
        self.activities_per_lesson = activities_per_lesson
        self.learning_styles = list(learning_styles)
        self.max_prerequisites = max_prerequisites
        self.max_lessons_per_activity = max_lessons_per_activity
        self.points = points

    # ───────────────────────────────────────────────────────── helpers ──
    @staticmethod
    def generate_lesson_id(idx: int) -> str:
        """
        Generates a unique, human-readable ID for a lesson based on its numerical index.
        For example, index 0 becomes "L1", index 1 becomes "L2", etc.

        Args:
            idx (int): The zero-based index of the lesson.

        Returns:
            str: The formatted lesson ID.
        """
        return f"L{idx+1}"

    @staticmethod
    def generate_activity_id(idx: int) -> str:
        """
        Generates a unique, human-readable ID for an activity based on its numerical index.
        For example, index 0 becomes "A1", index 1 becomes "A2", etc.

        Args:
            idx (int): The zero-based index of the activity.

        Returns:
            str: The formatted activity ID.
        """
        return f"A{idx+1}"

    def generate_learning_style(self) -> Dict[str, float]:
        """
        Generates a random VARK (Visual, Auditory, Read/Write, Kinesthetic) style vector.
        Each style receives a random probability, rounded to two decimal places.
        The probabilities are then normalized to ensure their sum is approximately 1.0.
        A loop ensures that the sum of probabilities is greater than zero to avoid division by zero.

        Returns:
            Dict[str, float]: A dictionary where keys are learning style names and values are
                              their normalized probability scores for an activity.
        """
        while True:
            vec = {s: round(random.random(), 2) for s in self.learning_styles}
            total = sum(vec.values())
            if total > 0:
                return {k: round(v / total, 2) for k, v in vec.items()}

    # ───────────────────────────────────────────────────── lessons ──
    def generate_lessons(self) -> List[Dict]:
        """
        Generates a list of lesson dictionaries. Each lesson has an 'id' and a
        'prerequisites' dictionary. The first lesson (L1) has no prerequisites.
        Subsequent lessons may have 1 to `max_prerequisites` randomly selected
        prerequisites from lessons that precede them. Each prerequisite is assigned
        a random mastery threshold between 0.4 and 0.8 that a student must meet.

        Returns:
            List[Dict]: A list of dictionaries, each representing a lesson.
        """
        lessons = []
        for i in range(self.num_lessons):
            lesson_id = self.generate_lesson_id(i)
            prerequisites = {}

            # Lesson 1 (index 0) has no prerequisites.
            if i:
                num = min(i, random.randint(1, self.max_prerequisites))
                for prereq in random.sample(
                    [self.generate_lesson_id(j) for j in range(i)], num
                ):
                    prerequisites[prereq] = round(random.uniform(0.4, 0.8), 2)

            lessons.append({"id": lesson_id, "prerequisites": prerequisites})
        return lessons

    # ───────────────────────────────────────────────── activity helpers ──
    def generate_activity_lessons(self, focal: str) -> Dict[str, float]:
        """
        Generates a dictionary specifying which lessons an activity covers and to what extent.
        The `focal` lesson is always included. The activity might also cover other lessons,
        up to `self.max_lessons_per_activity`. The coverage percentages are normalized
        to sum to 1.0. In some cases (1 out of `max_lessons_per_activity` chance),
        an activity will exclusively cover only the focal lesson.

        Args:
            focal (str): The ID of the primary lesson this activity is designed to cover.

        Returns:
            Dict[str, float]: A dictionary mapping lesson IDs to their coverage percentages
                              within this activity.
        """
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
                
                s = sum(coverage.values())
                if all(0 < v <= 1 for v in coverage.values()):
                    return {k: round(v / s, 2) for k, v in coverage.items()}

    # ────────────────────────────────────────────────── activities ──
    def generate_activities(self, lessons: List[Dict]) -> List[Dict]:
        """
        Generates a comprehensive list of activities for the curriculum.
        This includes:
        1. **Focused Activities**: `self.activities_per_lesson` activities are generated
           for each lesson, primarily focusing on that lesson but potentially covering others.
        2. **Mixed Review Activities**: An additional 10% of the total focused activities
           are generated as "mixed review," covering multiple lessons (from 2 up to
           `self.max_lessons_per_activity`) with balanced coverage.

        Args:
            lessons (List[Dict]): The list of lesson dictionaries generated previously.

        Returns:
            List[Dict]: A list of dictionaries, each representing a distinct activity.
        """
        acts: List[Dict] = []
        aid = 0  

        # 1. Generate focused activities for each lesson.
        for lesson in lessons:
            for _ in range(self.activities_per_lesson):
                acts.append(
                    {
                        "id": self.generate_activity_id(aid),
                        "lessons": self.generate_activity_lessons(lesson["id"]),
                        "style": self.generate_learning_style(),
                        "nb_points": self.points
                    }
                )
                aid += 1  
                
        # 2. Generate 10% mixed review activities.
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
                        break  # Break the inner loop once a valid activity is generated.
        return acts

    # ───────────────────────────────────────────── curriculum ──
    def generate_curriculum(self, file_path: str | None = None) -> Dict:
        """
        Orchestrates the generation of the complete learning curriculum.
        It first generates lessons, then activities based on those lessons,
        and finally compiles them into a single curriculum dictionary along with metadata.
        If a `file_path` is provided, the generated curriculum is saved as a JSON file.

        Args:
            file_path (str | None): Optional. The full path where the curriculum JSON file
                                     should be saved. If None, the curriculum is not saved.

        Returns:
            Dict: A dictionary representing the complete generated curriculum.
        """
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
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as fp:
                json.dump(curriculum, fp, indent=2)  
        return curriculum

if __name__ == "__main__":
    
    gen = CurriculumGenerator(num_lessons=4, activities_per_lesson=400)
    gen.generate_curriculum("data/learning_curriculum.json")
    print("Saved → data/learning_curriculum.json")