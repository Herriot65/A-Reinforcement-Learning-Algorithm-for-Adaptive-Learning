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
                 student: Student,
                 render_mode: Optional[str] = None,
                 max_steps: int = 1000,
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
        self.observation_space = spaces.Box(
            0.0, 1.0, (len(self.lessons) + len(self.activities),), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.activities))

        # Learning state
        self.lesson_mastery = LessonMasteryTracker(self.lessons)
        self.completed_activities = set()
        self.render_mode = render_mode

        # Student component
        self.student = student

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
        """
        Resets the environment to its initial state, preparing for a new episode.
        This includes resetting lesson mastery, clearing completed activities,
        and re-initializing the step count.
        If logging is enabled, it also saves the data from the previous episode.

        Args:
            seed (Optional[int]): A seed for the random number generator to ensure reproducibility.
            options (Optional[Dict]): Additional options for resetting the environment.

        Returns:
            Tuple[np.ndarray, Dict]: 
                - The initial observation of the environment.
                - An info dictionary containing available actions.
        """
        super().reset(seed=seed)

        # If there's data from the previous episode and logging is enabled, save it.
        if self.current_episode_data and self.log_training_data:
            self.training_data['episodes'].append({
                'episode': self.episode_count,
                'steps': self.current_episode_data.copy()
            })
            self._save_training_data()

        # Reset environment state for a new episode
        self.lesson_mastery.reset()
        self.completed_activities.clear()
        self.step_count = 0
        self.current_episode_data = []
        self.episode_count += 1

        # Log initial state of the new episode if logging is enabled
        if self.log_training_data:
            self.current_episode_data.append({
                'step': 0,
                'mastery_levels': self.lesson_mastery.mastery.copy(),
                'activity_id': None,
                'performance': 0.0
            })

        return self._get_state(), {'available_actions': self._get_available_actions_mask()}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Executes one learning step in the environment. The agent selects an activity
        (action), and the student's performance on that activity is simulated.
        The lesson mastery is updated, and the environment's state, reward,
        and termination conditions are returned.

        Args:
            action (int): The index of the activity chosen by the agent.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict]:
                - observation (np.ndarray): The new observation of the environment after the step.
                - reward (float): The reward received after performing the action (student's performance).
                - terminated (bool): Whether the episode has ended due to all lessons being mastered.
                - truncated (bool): Whether the episode has ended due to reaching the maximum number of steps.
                - info (Dict): A dictionary containing additional information, such as the activity ID,
                               performance, and available actions for the next step.
        """
        self.step_count += 1
        activity = self.activities[self.activity_ids[action]]
        self.completed_activities.add(activity['id'])

        performance = self.student.calculate_performance(activity)
        reward = performance

        self.lesson_mastery.update(activity, performance)

        # Log current step data if logging is enabled
        if self.log_training_data:
            self.current_episode_data.append({
                'step': self.step_count,
                'mastery_levels': self.lesson_mastery.mastery.copy(),
                'activity_id': activity['id'],
                'performance': performance
            })

        # Determine if the episode should terminate or truncate
        terminated = self.lesson_mastery.all_mastered()
        truncated = self.step_count >= self.max_steps

        return (
            self._get_state(),
            reward,
            terminated,
            truncated,
            {
                'activity_id': activity['id'],
                'performance': performance,
                'available_actions': self._get_available_actions_mask()
            }
        )

    def _get_available_actions_mask(self) -> np.ndarray:
        """
        Generates a binary mask indicating which activities are currently available to the agent.
        
        Returns:
            np.ndarray: A boolean array where `True` indicates an available action and `False` otherwise.
        """
        mask = np.zeros(len(self.activities), dtype=np.int8)
        for i, activity_id in enumerate(self.activity_ids):
            if activity_id in self.completed_activities:
                continue
            activity = self.activities[activity_id]
            mask[i] = int(self.lesson_mastery.is_activity_available(activity))
        return mask

    def _get_state(self) -> np.ndarray:
        """
        Constructs the current observation (state) of the environment.
        The state combines the current mastery levels of all lessons with a binary
        representation of completed activities.

        Returns:
            np.ndarray: A concatenated numpy array representing the current state.
        """
        mastery_array = self.lesson_mastery.to_array()
        activity_history = np.array([
            1.0 if activity_id in self.completed_activities else 0.0
            for activity_id in self.activity_ids
        ], dtype=np.float32)
        return np.concatenate([mastery_array, activity_history])

    def _save_training_data(self):
        """
        Saves the accumulated training data for all episodes to the specified JSON file.
        """
        try:
            os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else '.', exist_ok=True)
            with open(self.log_file, 'w') as f:
                json.dump(self.training_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save training data: {e}")

    def finalize_logging(self):
        """
        Capture the data from the last episode.
        """
        if self.current_episode_data and self.log_training_data:
            self.training_data['episodes'].append({
                'episode': self.episode_count,
                'steps': self.current_episode_data.copy()
            })
            self._save_training_data()
            print(f"Training data saved to {self.log_file}")
            print(f"Total episodes logged: {len(self.training_data['episodes'])}")