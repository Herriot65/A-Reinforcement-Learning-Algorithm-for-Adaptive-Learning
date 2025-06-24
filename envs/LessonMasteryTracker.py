import numpy as np
from typing import Dict

class LessonMasteryTracker:
    """Encapsulates lesson mastery logic."""

    def __init__(self, lessons: Dict[str, Dict]):
        """
        Initializes the LessonMasteryTracker.

        Args:
            lessons (Dict[str, Dict]): A dictionary where keys are lesson IDs and
                                       values are dictionaries containing lesson details,
                                       including 'prerequisites' if any.
        """
        self.lessons = lessons
        self.mastery = {l_id: 0.0 for l_id in lessons}

    def reset(self):
        """
        Resets all lesson mastery levels to 0.0.
        """
        self.mastery = {l_id: 0.0 for l_id in self.lessons}

    def update(self, activity: Dict, performance: float):
        """
        Updates the mastery levels for the lessons covered by a given activity.
        Mastery increases based on the student's performance and the lesson's coverage
        within the activity, capped at 1.0.

        Args:
            activity (Dict): The activity dictionary, which must contain a 'lessons' key
                             mapping lesson IDs to their coverage (e.g., {'lesson_A': 0.5}).
            performance (float): The student's performance score on the activity (0.0 to 1.0).
        """
        for lesson_id, cov in activity['lessons'].items():
            # Only update if the lesson is not yet fully mastered
            if self.mastery[lesson_id] < 1.0:
                # Mastery increases by a fraction of performance * coverage (scaled down by 0.01)
                self.mastery[lesson_id] = min(1.0, self.mastery[lesson_id] + performance * cov * 0.01)

    def all_mastered(self) -> bool:
        """
        Checks if all lessons in the curriculum have reached full mastery (>= 1.0).

        Returns:
            bool: True if all lessons are mastered, False otherwise.
        """
        return all(m >= 1.0 for m in self.mastery.values())

    def is_activity_available(self, activity: Dict) -> bool:
        """
        Determines if an activity is currently available for the student.
        An activity is available if:
        1. It covers at least one lesson that the student has NOT yet mastered.
        2. For every lesson covered by the activity, all of that lesson's prerequisites
           (if any) must have reached their required mastery threshold.

        Args:
            activity (Dict): The activity dictionary, containing 'id', 'lessons' (coverage),
                             and other details.

        Returns:
            bool: True if the activity meets all availability criteria, False otherwise.
        """
        has_unmastered = False
        for lesson_id, _ in activity['lessons'].items():
            if self.mastery[lesson_id] == 1.0:
                continue
            has_unmastered = True
            
            #get lesson's prerequisites
            prerequisites = self.lessons[lesson_id].get('prerequisites', {})
            
            #if lesson has prereqs, check if they are mastered
            if prerequisites:
                for prereq, req_m in prerequisites.items():
                    if self.mastery[prereq] < req_m:
                        return False
        return has_unmastered

    def to_array(self) -> np.ndarray:
        """
        Converts the current mastery levels of all lessons into a NumPy array.
        The order of lessons in the array corresponds to the order of lesson IDs
        as they appear in the `self.lessons` dictionary .

        Returns:
            np.ndarray: A 1D float32 NumPy array representing the mastery levels.
        """
        return np.array(list(self.mastery.values()), dtype=np.float32)