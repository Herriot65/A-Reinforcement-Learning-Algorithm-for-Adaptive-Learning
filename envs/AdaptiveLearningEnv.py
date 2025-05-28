import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from . import Student  
from .LessonMasteryTracker import LessonMasteryTracker

class AdaptiveLearningEnv(gym.Env):
    """Gymnasium environment with integrated Student class and training data logging."""
    
    metadata = {'render_modes': ['human'], 'render_fps': 4}
    
    def __init__(self, 
                 lessons: List[Dict], 
                 activities: List[Dict],
                 student: Student,  # Use Student instance
                 render_mode: Optional[str] = None,
                 max_steps: int = 150,
                 log_training_data: bool = True,
                 log_file: str = "training_data.json"):
        super().__init__()
        
        # Curriculum data
        self.lessons = {l['id']: l for l in lessons}
        self.activities = {a['id']: a for a in activities}
        self.activity_ids = list(self.activities.keys())
        
        # RL settings
        self.max_steps = max_steps
        self.step_count = 0
        self.observation_space = spaces.Box(0.0, 1.0, (len(self.lessons),), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.activities))
        
        # Learning state
        self.lesson_mastery = LessonMasteryTracker(self.lessons)  # Separated logic
        self.completed_activities = set()
        self.render_mode = render_mode
        
        # Student component
        self.student = student
        
        self.ALIGNMENT_BONUS_WEIGHT = 0.5
        
        # Training data logging
        self.log_training_data = log_training_data
        self.log_file = log_file
        self.training_data = {
            'episodes': [],
            'student_profile': {
                'dominant_style': student.dominant_style,
                'dominant_percent': student.dominant_percent,
                'learning_style': student.learning_style
            },
            'lessons': list(self.lessons.keys())
        }
        self.current_episode_data = []
        self.episode_count = 0

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment with new episode."""
        super().reset(seed=seed)
        
        # Save previous episode data if exists
        if self.current_episode_data and self.log_training_data:
            self.training_data['episodes'].append({
                'episode': self.episode_count,
                'steps': self.current_episode_data.copy()
            })
            self._save_training_data()
        
        # Reset for new episode
        self.lesson_mastery.reset()
        self.completed_activities.clear()
        self.step_count = 0
        self.current_episode_data = []
        self.episode_count += 1
        
        # Log initial state
        if self.log_training_data:
            self.current_episode_data.append({
                'step': 0,
                'mastery_levels': self.lesson_mastery.mastery.copy(),
                'activity_id': None,
                'performance': 0.0
            })
        
        return self._get_state(), {'available_actions': self._get_available_actions_mask()}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one learning step."""
        self.step_count += 1
        activity = self.activities[self.activity_ids[action]]
        self.completed_activities.add(activity['id'])

        # Student performance calculation
        performance = self.student.calculate_performance(activity)
        
        # Reward calculation
        reward = performance
        
        # Mastery update
        self.lesson_mastery.update(activity, performance)
        
        # Log training data
        if self.log_training_data:
            self.current_episode_data.append({
                'step': self.step_count,
                'mastery_levels': self.lesson_mastery.mastery.copy(),
                'activity_id': activity['id'],
                'performance': performance
            })
        
        # Termination conditions
        terminated = self.lesson_mastery.all_mastered()
        truncated = self.step_count >= self.max_steps
        
        return (
            self._get_state(),
            reward,  # Reward
            terminated,
            truncated,
            {
                'activity_id': activity['id'],
                'performance': performance,
                'available_actions': self._get_available_actions_mask()
            }
        )

    def _get_available_actions_mask(self) -> np.ndarray:
        """Generate action mask based on prerequisites/mastery."""
        mask = np.zeros(len(self.activities), dtype=np.int8)
        for i, activity_id in enumerate(self.activity_ids):
            if activity_id in self.completed_activities:
                continue
            activity = self.activities[activity_id]
            mask[i] = int(self.lesson_mastery.is_activity_available(activity))
        return mask

    def _get_state(self) -> np.ndarray:
        """Get current state as numpy array."""
        return self.lesson_mastery.to_array()
    
    def _save_training_data(self):
        """Save training data to JSON file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else '.', exist_ok=True)
            
            with open(self.log_file, 'w') as f:
                json.dump(self.training_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save training data: {e}")
    
    def finalize_logging(self):
        """Call this at the end of training to save final episode data."""
        if self.current_episode_data and self.log_training_data:
            self.training_data['episodes'].append({
                'episode': self.episode_count,
                'steps': self.current_episode_data.copy()
            })
            self._save_training_data()
            print(f"Training data saved to {self.log_file}")
            print(f"Total episodes logged: {len(self.training_data['episodes'])}")