import os
import sys
import json
import random
from typing import Dict, List

# Add the parent directory to the system path. This is a common practice
# to allow importing modules from higher-level directories in a project
# structure, especially when running scripts from a subdirectory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CurriculumGenerator:
    """
    Creates a synthetic learning curriculum in JSON format for an adaptive-learning
    Reinforcement Learning (RL) simulation environment.

    This generator customizes activities with detailed attributes:
        • lessons     : A dictionary where keys are lesson IDs and values represent
                        the `coverage` of that lesson within the activity (a float between 0 and 1).
                        This indicates how much an activity contributes to understanding a specific lesson.
        • style       : A dictionary representing the activity's alignment with VARK learning styles
                        (Visual, Auditory, Read/Write, Kinesthetic). Values are floats summing to 1,
                        indicating the proportion of each style addressed.
        • nb_points   : An integer representing the `points` awarded to a student
                        if they successfully complete this activity.
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
        # Convert to list to allow modification if needed, though not used here.
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
            # Assign a random value between 0 and 1 (rounded to 2 decimals) for each style.
            vec = {s: round(random.random(), 2) for s in self.learning_styles}
            total = sum(vec.values())
            if total > 0:
                # Normalize the values so they sum up to 1.0.
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
                # Determine a random number of prerequisites for the current lesson,
                # ensuring it doesn't exceed the number of already generated lessons.
                num = min(i, random.randint(1, self.max_prerequisites))
                # Randomly select `num` lesson IDs from the lessons generated so far (0 to i-1).
                for prereq in random.sample(
                    [self.generate_lesson_id(j) for j in range(i)], num
                ):
                    # Assign a random mastery threshold for each prerequisite.
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
        # A simple case: the activity focuses entirely on the focal lesson.
        if random.randint(1, self.max_lessons_per_activity) == 1:
            return {focal: 1.0}

        # Otherwise, the activity covers the focal lesson significantly and potentially others.
        main_cov = round(random.uniform(0.5, 0.9), 2)  # High coverage for the focal lesson.
        # Get all lesson IDs except the focal one.
        others = [self.generate_lesson_id(i) for i in range(self.num_lessons) if self.generate_lesson_id(i) != focal]
        # Randomly decide how many other lessons (besides focal) to include.
        n_others = random.randint(1, self.max_lessons_per_activity - 1)
        # Select specific other lessons to include.
        selected = random.sample(others, n_others)

        while True:
            # Assign random weights to the selected other lessons.
            weights = [round(random.random(), 2) for _ in selected]
            tot = sum(weights)
            if tot:  # Ensure total weight is not zero to prevent division by zero.
                coverage = {focal: main_cov}
                rem = round(1.0 - main_cov, 2)  # Remaining coverage to distribute.
                # Distribute the remaining coverage proportionally among other selected lessons.
                for w, les in zip(weights, selected):
                    coverage[les] = round(rem * w / tot, 2)
                
                # Re-normalize the entire coverage dictionary to ensure sum is exactly 1.0
                # after rounding errors, and all values are positive.
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
        aid = 0  # Initialize activity ID counter.

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
                aid += 1  # Increment activity ID for the next activity.

        # 2. Generate 10% mixed review activities.
        # These activities cover multiple lessons, not primarily focused on one.
        for _ in range(int(len(acts) * 0.1)):
            # Randomly select 2 to `self.max_lessons_per_activity` lesson IDs for mixed review.
            select = random.sample([l["id"] for l in lessons],
                                   random.randint(2, self.max_lessons_per_activity))
            while True:
                # Generate random weights for each selected lesson.
                w = [round(random.random(), 2) for _ in select]
                # Ensure the sum of weights is positive.
                if (tot := sum(w)):
                    # Calculate coverage for each selected lesson based on its weight.
                    cov = {l: round(v / tot, 2) for l, v in zip(select, w)}
                    # Ensure the total coverage sums to approximately 1.0 (handle floating point inaccuracies).
                    if abs(sum(cov.values()) - 1.0) < 1e-3:
                        acts.append(
                            {
                                "id": self.generate_activity_id(aid),
                                "lessons": cov,
                                "style": self.generate_learning_style(),
                                "nb_points": self.points,
                            }
                        )
                        aid += 1  # Increment activity ID.
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
            # Ensure the directory for the file exists.
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as fp:
                json.dump(curriculum, fp, indent=2)  # Save with a 2-space indentation for readability.
        return curriculum

if __name__ == "__main__":
    
    # Create an instance of the CurriculumGenerator with 4 lessons and 400 activities per lesson.
    gen = CurriculumGenerator(num_lessons=4, activities_per_lesson=400)
    
    # Generate the curriculum and save it to the specified JSON file.
    gen.generate_curriculum("data/learning_curriculum.json")
    print("Saved → data/learning_curriculum.json")