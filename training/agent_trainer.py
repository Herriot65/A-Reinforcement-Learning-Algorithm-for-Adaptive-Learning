import os
import sys 
import json
import config
import torch.distributions
from sb3_contrib.ppo_mask import MaskablePPO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.Student import Student
from tests.student_model_tester import ModelTester
from .environment_setup import load_curriculum, create_vec_env
from tests.learning_progress_analyzer import analyze_learning_progress

# Ignore warnings
torch.distributions.Distribution._validate_args = False

def train_single_student(
    student,
    curriculum_file=config.CURRICULUM_FILE,
    models_dir=config.MODELS_DIR,
    logs_dir=config.LOGS_DIR,
    data_dir=config.DATA_DIR,
    total_timesteps=config.DEFAULT_TOTAL_TIMESTEPS,
    n_envs=config.N_ENVS,
    log_file_name_json=config.LOG_FILE_NAME_JSON,
    model_name=config.MODEL_NAME,
    learning_rate=config.LEARNING_RATE,
    batch_size=config.BATCH_SIZE,
    n_epochs=config.N_EPOCHS,
    clip_range=config.CLIP_RANGE,
    n_steps=config.N_STEPS,
    gamma=config.GAMMA,
    gae_lambda=config.GAE_LAMBDA,
    ent_coef=config.ENT_COEF,
    vf_coef=config.VF_COEF,
    max_grad_norm=config.MAX_GRAD_NORM,
    verbose=config.VERBOSE,
    test_after_training=config.TEST_AFTER_TRAINING,
    test_episodes=config.TEST_EPISODES,
    analyze_progress=config.ANALYZE_PROGRESS,
    max_steps=config.MAX_STEPS_PER_EPISODE
):
    """
    Trains a single PPO (Proximal Policy Optimization) agent for a given student
    model within the adaptive learning environment.

    This function handles the entire training pipeline: environment creation,
    agent initialization, training, model saving, and optional post-training
    testing and learning progress analysis.

    Args:
        student (Student): An instance of the Student class representing the
                           student for whom the model is being trained.
        curriculum_file (str, optional): Path to the JSON file defining the curriculum.
                                         Defaults to config.CURRICULUM_FILE.
        models_dir (str, optional): Directory to save the trained models.
                                    Defaults to config.MODELS_DIR.
        logs_dir (str, optional): Directory for TensorBoard logs during training.
                                  Defaults to config.LOGS_DIR.
        data_dir (str, optional): Directory to save training data logs.
                                  Defaults to config.DATA_DIR.
        total_timesteps (int, optional): Total number of environmental steps to train the agent.
                                         Defaults to config.DEFAULT_TOTAL_TIMESTEPS.
        n_envs (int, optional): Number of parallel environments to run for training.
                                Defaults to config.N_ENVS.
        log_file_name_json (str, optional): Base name for the JSON log file storing training data.
                                            A student-specific ID will be appended to this.
                                            Defaults to config.LOG_FILE_NAME_JSON.
        model_name (str, optional): Base name for the trained model file.
                                    A student-specific ID will be appended to this.
                                    Defaults to config.MODEL_NAME.
        learning_rate (float, optional): Learning rate for the PPO optimizer.
                                         Defaults to config.LEARNING_RATE.
        batch_size (int, optional): Mini-batch size for PPO updates.
                                    Defaults to config.BATCH_SIZE.
        n_epochs (int, optional): Number of epochs for each PPO update.
                                  Defaults to config.N_EPOCHS.
        clip_range (float, optional): Clipping parameter for PPO.
                                      Defaults to config.CLIP_RANGE.
        n_steps (int, optional): The number of steps to run for each environment per update.
                                 Defaults to config.N_STEPS.
        gamma (float, optional): Discount factor. Defaults to config.GAMMA.
        gae_lambda (float, optional): Factor for trade-off of bias vs variance for GAE.
                                      Defaults to config.GAE_LAMBDA.
        ent_coef (float, optional): Entropy coefficient for the loss function.
                                    Defaults to config.ENT_COEF.
        vf_coef (float, optional): Value function coefficient for the loss function.
                                   Defaults to config.VF_COEF.
        max_grad_norm (float, optional): The maximum value for the gradient clipping.
                                         Defaults to config.MAX_GRAD_NORM.
        verbose (int, optional): Verbosity level (0: no output, 1: info, 2: debug).
                                 Defaults to config.VERBOSE.
        test_after_training (bool, optional): Whether to run a test on the trained model.
                                             Defaults to config.TEST_AFTER_TRAINING.
        test_episodes (int, optional): Number of episodes to run during testing.
                                       Defaults to config.TEST_EPISODES.
        analyze_progress (bool, optional): Whether to analyze the learning progress from the log file.
                                          Defaults to config.ANALYZE_PROGRESS.
        max_steps (int, optional): Maximum steps per episode during training.
                                   Defaults to config.MAX_STEPS_PER_EPISODE.

    Returns:
        dict: A dictionary containing paths to the generated log file, saved model,
              output directory for test results, and the student's ID.
    """
    lessons, activities = load_curriculum(curriculum_file)

    # Create a unique ID for the student based on their characteristics
    student_id = f"{student.dominant_style}_{student.dominant_percent}pct_{str(student.velocity).replace('.', '')}vel".replace("/", "_")

    log_file = os.path.join(data_dir, f"{log_file_name_json.replace('.json', '')}_{student_id}.json")
    model_path = os.path.join(models_dir, f"{model_name}_{student_id}")
    unified_output_dir = os.path.join(config.MODEL_TEST_RESULTS_BASE_DIR, f"{model_name}_{student_id}/")
    os.makedirs(unified_output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True) 

    # Create vectorized environment
    train_env = create_vec_env(n_envs, lessons, activities, log_file, student, max_steps)

    # Initialize and train PPO agent
    print(f"Starting training for student: {student_id}")
    agent = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=verbose,
        tensorboard_log=f"{logs_dir}/{student_id}",
    )

    agent.learn(total_timesteps=total_timesteps)
    agent.save(model_path)
    print(f"Model saved to: {model_path}")

    # Test the trained model
    if test_after_training:
        print(f"Testing trained model for {student_id}...")
        try:
            tester = ModelTester(
                model_path=model_path,
                curriculum_path=curriculum_file,
                output_dir=unified_output_dir,
                mastery_threshold=config.MASTERY_THRESHOLD 
            )
            _ = tester.test_student(student, num_episodes=test_episodes, student_name=f"{student_id}_trained")

        except Exception as e:
            print(f"Error during model testing: {e}")

    if analyze_progress:
        print(f"Analyzing learning progress for {student_id}...")
        try:
            analyze_learning_progress(log_file, unified_output_dir)
        except Exception as e:
            print(f"Error during learning progress analysis: {e}")

    metadata = {
        "algorithm": "PPO",
        "total_timesteps_trained": total_timesteps,
        "model_name": model_name,
        "n_envs": n_envs,
        "max_steps": max_steps,
        "curriculum_file": curriculum_file,
        "hyperparameters": {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm
        },
        "student_config": {
            "dominant_style": student.dominant_style,
            "dominant_percent": student.dominant_percent,
            "velocity": student.velocity
        }
    }

    with open(f"{unified_output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "log_file": log_file,
        "model_path": model_path,
        "output_dir": unified_output_dir,
        "student_id": student_id
    }

def train(
    students=None,
    curriculum_file=config.CURRICULUM_FILE,
    models_dir=config.MODELS_DIR,
    logs_dir=config.LOGS_DIR,
    data_dir=config.DATA_DIR,
    total_timesteps=config.DEFAULT_TOTAL_TIMESTEPS,
    n_envs=config.N_ENVS,
    log_file_name_json=config.LOG_FILE_NAME_JSON,
    model_name=config.MODEL_NAME,
    learning_rate=config.LEARNING_RATE,
    batch_size=config.BATCH_SIZE,
    n_epochs=config.N_EPOCHS,
    clip_range=config.CLIP_RANGE,
    n_steps=config.N_STEPS,
    gamma=config.GAMMA,
    gae_lambda=config.GAE_LAMBDA,
    ent_coef=config.ENT_COEF,
    vf_coef=config.VF_COEF,
    max_grad_norm=config.MAX_GRAD_NORM,
    verbose=config.VERBOSE,
    test_after_training=config.TEST_AFTER_TRAINING,
    test_episodes=config.TEST_EPISODES,
    analyze_progress=config.ANALYZE_PROGRESS,
):
    """
    Orchestrates the training of PPO agents for one or more student models.

    This function sets up the necessary directories, iterates through a list
    of student configurations, and calls `train_single_student` for each.
    It also adjusts `total_timesteps` for students with lower velocities.

    Args:
        students (list[Student] or Student, optional): A single Student object or a list
                                                        of Student objects for whom to train models.
                                                        If None, a default student is used.
                                                        Defaults to None.
        curriculum_file (str, optional): Path to the JSON file defining the curriculum.
                                         Defaults to config.CURRICULUM_FILE.
        models_dir (str, optional): Directory to save the trained models.
                                    Defaults to config.MODELS_DIR.
        logs_dir (str, optional): Directory for TensorBoard logs during training.
                                  Defaults to config.LOGS_DIR.
        data_dir (str, optional): Directory to save training data logs.
                                  Defaults to config.DATA_DIR.
        total_timesteps (int, optional): Total number of environmental steps to train
                                         each agent (can be overridden for slower students).
                                         Defaults to config.DEFAULT_TOTAL_TIMESTEPS.
        n_envs (int, optional): Number of parallel environments to run for training.
                                Defaults to config.N_ENVS.
        log_file_name_json (str, optional): Base name for the JSON log file storing
                                            training data. Defaults to config.LOG_FILE_NAME_JSON.
        model_name (str, optional): Base name for the trained model file.
                                    Defaults to config.MODEL_NAME.
        learning_rate (float, optional): Learning rate for the PPO optimizer.
                                         Defaults to config.LEARNING_RATE.
        batch_size (int, optional): Mini-batch size for PPO updates.
                                    Defaults to config.BATCH_SIZE.
        n_epochs (int, optional): Number of epochs for each PPO update.
                                  Defaults to config.N_EPOCHS.
        clip_range (float, optional): Clipping parameter for PPO.
                                      Defaults to config.CLIP_RANGE.
        n_steps (int, optional): The number of steps to run for each environment per update.
                                 Defaults to config.N_STEPS.
        gamma (float, optional): Discount factor. Defaults to config.GAMMA.
        gae_lambda (float, optional): Factor for trade-off of bias vs variance for GAE.
                                      Defaults to config.GAE_LAMBDA.
        ent_coef (float, optional): Entropy coefficient for the loss function.
                                    Defaults to config.ENT_COEF.
        vf_coef (float, optional): Value function coefficient for the loss function.
                                   Defaults to config.VF_COEF.
        max_grad_norm (float, optional): The maximum value for the gradient clipping.
                                         Defaults to config.MAX_GRAD_NORM.
        verbose (int, optional): Verbosity level (0: no output, 1: info, 2: debug).
                                 Defaults to config.VERBOSE.
        test_after_training (bool, optional): Whether to run a test on the trained model.
                                             Defaults to config.TEST_AFTER_TRAINING.
        test_episodes (int, optional): Number of episodes to run during testing.
                                       Defaults to config.TEST_EPISODES.
        analyze_progress (bool, optional): Whether to analyze the learning progress from the log file.
                                          Defaults to config.ANALYZE_PROGRESS.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains the
                    training results (log file, model path, output directory, student ID)
                    for each trained student.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(config.MODEL_TEST_RESULTS_BASE_DIR, exist_ok=True)

    if students is None:
        students = [Student(dominant_style="Visual", dominant_percent=80, velocity=0.65)]
    elif not isinstance(students, list):
        students = [students]

    results = []
    for student in students:
        current_total_timesteps = total_timesteps
        if student.velocity <= config.SLOW_STUDENT_VELOCITY_THRESHOLD:
            current_total_timesteps = config.SLOW_STUDENT_TOTAL_TIMESTEPS

        result = train_single_student(
            student=student,
            curriculum_file=curriculum_file,
            models_dir=models_dir,
            logs_dir=logs_dir,
            data_dir=data_dir,
            total_timesteps=current_total_timesteps,
            n_envs=n_envs,
            log_file_name_json=log_file_name_json,
            model_name=model_name,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_epochs=n_epochs,
            clip_range=clip_range,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
            test_after_training=test_after_training,
            test_episodes=test_episodes,
            analyze_progress=analyze_progress,
            max_steps=config.MAX_STEPS_PER_EPISODE
        )
        results.append(result)

    return results