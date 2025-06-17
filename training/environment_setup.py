import json
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.wrappers import ActionMasker

import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.AdaptiveLearningEnv import AdaptiveLearningEnv

def mask_fn(env):
    """
    Returns the action mask from the environment. This mask indicates which actions
    are currently valid (e.g., based on prerequisites or already completed activities).
    This function is used by the ActionMasker wrapper for Stable Baselines3.

    Args:
        env (gym.Env): The Gymnasium environment instance.

    Returns:
        np.ndarray: A binary array where 1 indicates a valid action and 0 an invalid one.
    """
    return env._get_available_actions_mask()

def load_curriculum(curriculum_path):
    """
    Loads the learning curriculum from a specified JSON file.

    The JSON file is expected to contain "lessons" and "activities" keys,
    each mapping to a list of relevant curriculum components.

    Args:
        curriculum_path (str): The file path to the JSON curriculum definition.

    Returns:
        tuple: A tuple containing two elements:
               - lessons (list): A list of lesson definitions.
               - activities (list): A list of activity definitions.
    """
    with open(curriculum_path) as f:
        data = json.load(f)
    return data["lessons"], data["activities"]

def make_adaptive_learning_env(lessons, activities, log_file, student, max_steps):
    """
    Creates a function that initializes and returns a wrapped AdaptiveLearningEnv.

    This function is designed to be used with Stable Baselines3's `make_vec_env`
    to create multiple instances of the environment, each wrapped with an
    `ActionMasker` to handle valid actions.

    Args:
        lessons (list): A list of lesson definitions for the environment.
        activities (list): A list of activity definitions for the environment.
        log_file (str): The file path where training data will be logged.
        student (Student): An instance of the Student class, representing the
                           student model interacting with the environment.
        max_steps (int): The maximum number of steps (actions) allowed
                         in a single episode.

    Returns:
        callable: A function that, when called, returns an instance of
                  ActionMasker(AdaptiveLearningEnv).
    """
    def _init():
        env = AdaptiveLearningEnv(
            lessons,
            activities,
            student=student,
            log_training_data=True,
            log_file=log_file,
            max_steps=max_steps
        )
        return ActionMasker(env, mask_fn)
    return _init

def create_vec_env(n_envs, lessons, activities, log_file, student, max_steps):
    """
    Creates a vectorized environment for training, wrapping the adaptive
    learning environment with action masking.

    Args:
        n_envs (int): The number of parallel environments to create.
        lessons (list): Lesson definitions for the environment.
        activities (list): Activity definitions for the environment.
        log_file (str): Log file path for environment data.
        student (Student): The student model for the environment.
        max_steps (int): Maximum steps per episode.

    Returns:
        stable_baselines3.common.vec_env.VecEnv: A vectorized environment.
    """
    return make_vec_env(
        make_adaptive_learning_env(lessons, activities, log_file, student=student, max_steps=max_steps),
        n_envs=n_envs
    )