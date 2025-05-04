import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AdaptiveLearningEnv(gym.Env):
    """
    A custom Gym environment for simulating an adaptive learning system.
    The environment models a learner progressing through educational activities
    and lessons over multiple sprints, based on mastery, learning style, and engagement.

    Attributes:
        lessons (dict): Dictionary of lesson data with prerequisites and required mastery.
        activities (dict): Dictionary of activity data linked to lessons and difficulty.
        sprints (dict): Mapping of sprint IDs to lesson groupings.
        max_activities (int): Maximum steps before the episode terminates.
    """

    def __init__(self, lessons, activities, sprints, max_activities=100):
        """
        Initialize the learning environment.

        Args:
            lessons (dict): All available lessons with prerequisites and requirements.
            activities (dict): Activities with associated lessons, difficulty, and style.
            sprints (dict): A mapping of sprint IDs to associated lessons.
            max_activities (int): Maximum number of activities allowed per episode.
        """
        super().__init__()
        self.lessons = lessons
        self.activities = activities
        self.sprints = sprints
        self.max_activities = max_activities
        self.current_sprint_id = 1

        self.steps_taken = 0

        self.n_lessons = len(self.lessons)
        self.n_activities = len(self.activities)
        self.state_dim = self.n_lessons + 1 + 3 + self.n_activities  # mastery + velocity + style + action_mask
        self.action_space = spaces.Discrete(self.n_activities)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        """
        Resets the environment for a new episode.

        Args:
            seed (int, optional): Seed for reproducibility.
            options (dict, optional): Additional options.

        Returns:
            obs (np.ndarray): Initial observation state.
            info (dict): Additional information (empty here).
        """
        super().reset(seed=seed)
        self.mastery = {lid: 0.0 for lid in self.lessons}
        self.velocity = np.clip(np.random.normal(0.6, 0.1), 0.4, 0.9)
        self.style = np.random.choice(['visual', 'auditory', 'kinesthetic'])
        self.style_one_hot = self._encode_style(self.style)
        self.usage_counter = {aid: 0 for aid in self.activities}
        self.steps_taken = 0
        self.current_sprint_id = 1
        obs = self._get_state()
        info = {}
        return obs, info

    def _encode_style(self, style):
        """
        One-hot encode the learner's preferred style.

        Args:
            style (str): One of 'visual', 'auditory', or 'kinesthetic'.

        Returns:
            list[int]: One-hot encoded style vector.
        """
        return [int(style == s) for s in ['visual', 'auditory', 'kinesthetic']]

    def _get_state(self):
        """
        Builds the current observation state vector.

        Returns:
            np.ndarray: Concatenated mastery levels, velocity, style encoding, and action mask.
        """
        mastery_values = [self.mastery[lid] for lid in self.lessons]
        mask = self.get_action_mask().astype(np.float32)
        return np.array(mastery_values + [self.velocity] + self.style_one_hot + list(mask), dtype=np.float32)

    def _prerequisites_satisfied(self, lesson_id):
        """
        Checks whether the prerequisites for a lesson are satisfied.

        Args:
            lesson_id (str): Lesson ID to check.

        Returns:
            bool: True if all prerequisites are met, False otherwise.
        """
        prerequisites = self.lessons[lesson_id]['prerequisites']
        for prereq in prerequisites:
            for pre_lesson, required_level in prereq.items():
                if self.mastery.get(pre_lesson, 0.0) < required_level:
                    return False
        return True

    def _unlocked_new_lessons(self, improved_lesson):
        """
        Determines if improving a lesson unlocks any new lessons.

        Args:
            improved_lesson (str): The lesson that was just improved.

        Returns:
            bool: True if any new lesson got unlocked.
        """
        for l_id, lesson_data in self.lessons.items():
            for prereq in lesson_data['prerequisites']:
                if improved_lesson in prereq:
                    required_level = prereq[improved_lesson]
                    if self.mastery[improved_lesson] >= required_level:
                        return True
        return False

    def _get_current_sprint_activities(self):
        """
        Gets the list of valid activities for the current sprint.

        Returns:
            list[str]: Activity IDs that are relevant to the current sprint.
        """
        sprint_lessons = set(self.sprints[self.current_sprint_id]["lessons"])
        valid_activities = []
        for aid, activity in self.activities.items():
            if any(lid in sprint_lessons for lid in activity["lesson_contributions"]):
                valid_activities.append(aid)
        return valid_activities

    def get_action_mask(self):
        """
        Computes a boolean mask of valid activity actions for the current step.

        Returns:
            np.ndarray: Boolean array where True marks a valid action.
        """
        valid_ids = self._get_current_sprint_activities()
        valid_indices = [i for i, aid in enumerate(self.activities.keys()) if aid in valid_ids]
        mask = np.zeros(len(self.activities), dtype=np.bool_)
        mask[valid_indices] = True
        return mask

    def step(self, action):
        """
        Executes an action (selects an activity) and updates the environment.

        Args:
            action (int): Index of the selected activity.

        Returns:
            obs (np.ndarray): Updated observation state.
            reward (float): Reward obtained from the action.
            done (bool): True if episode is complete.
            truncated (bool): True if episode was truncated (max steps reached).
            info (dict): Additional information, including invalid action flag.
        """
        reward = 0.0
        info = {}
        done_flag = False

        # Increment step counter early to correctly handle truncation
        self.steps_taken += 1

        # Check if episode should be truncated due to max steps
        truncated = self.steps_taken > self.max_activities
        if truncated:
            return self._get_state(), 0.0, True, True, {"reason": "max_steps_exceeded"}

        # Get the activity ID corresponding to the action index
        activity_id = list(self.activities.keys())[action]
        # Retrieve the activity's metadata
        activity = self.activities[activity_id]

        # Get valid activities based on the current sprint
        valid_activities = self._get_current_sprint_activities()
        # Penalize if the selected activity is not valid for this sprint
        if activity_id not in valid_activities:
            return self._get_state(), -1.0, False, False, {"invalid_action": True}

        # Penalize if the activity has been selected more than allowed
        if self.usage_counter[activity_id] >= activity['max_selection_limit']:
            reward -= 0.2
        else:
            self.usage_counter[activity_id] += 1

            difficulty_weight = {'easy': 0.8, 'medium': 1.0, 'hard': 1.2}[activity['difficulty']]
            performance_score = np.clip(np.random.normal(0.8, 0.05), 0.0, 1.0)

            for lid, base_contribution in activity['lesson_contributions'].items():
                if self._prerequisites_satisfied(lid):
                    gain = round(float(base_contribution * performance_score * difficulty_weight * self.velocity), 3)
                    old_mastery = self.mastery[lid]
                    self.mastery[lid] = min(1.0, self.mastery[lid] + gain)

                    reward += 2.0 * gain

                    if (self.lessons[lid]['required_mastery_level'] > old_mastery and
                            self.mastery[lid] >= self.lessons[lid]['required_mastery_level']):
                        reward += 2.0

                    if old_mastery < 1.0 and self.mastery[lid] == 1.0:
                        reward += 3.0

                    if self._unlocked_new_lessons(lid):
                        reward += 2.5

            if activity['style'] == self.style:
                reward += 0.5

        # Check if all lessons in the current sprint are mastered
        all_mastered = all(
            self.mastery[lid] >= self.lessons[lid]["required_mastery_level"]
            for lid in self.sprints[self.current_sprint_id]["lessons"]
        )

        if all_mastered and self.current_sprint_id < max(self.sprints.keys()):
            self.current_sprint_id += 1
            reward += 5.0

        all_sprints_completed = (
            self.current_sprint_id == max(self.sprints.keys()) and
            all_mastered
        )

        done_flag = all_sprints_completed or truncated

        obs = self._get_state()
        return obs, reward, done_flag, truncated, info





