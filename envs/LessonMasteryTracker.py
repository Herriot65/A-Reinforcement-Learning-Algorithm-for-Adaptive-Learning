from typing import Dict
import numpy as np

class LessonMasteryTracker:
    """Encapsulates lesson mastery logic."""
    
    def __init__(self, lessons: Dict[str, Dict]):
        self.lessons = lessons
        self.mastery = {l_id: 0.0 for l_id in lessons}
    
    def reset(self):
        """Reset all mastery levels."""
        self.mastery = {l_id: 0.0 for l_id in self.lessons}
    
    def update(self, activity: Dict, performance: float):
        """Update mastery for activity's lessons."""
        for lesson_id, cov in activity['lessons'].items():
            if self.mastery[lesson_id] < 1.0:
                self.mastery[lesson_id] = min(1.0, self.mastery[lesson_id] + performance * cov * 0.01)
    
    def all_mastered(self) -> bool:
        """Check if all lessons are mastered."""
        return all(m >= 1.0 for m in self.mastery.values())
    
    def is_activity_available(self, activity: Dict) -> bool:
        """Check if activity is available (prerequisites/mastery)."""
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
        """Convert mastery to numpy array."""
        return np.array(list(self.mastery.values()), dtype=np.float32)